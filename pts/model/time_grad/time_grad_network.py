from torch.nn.modules import loss
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import TimeSeriesTransformerModel, TimeSeriesTransformerConfig

from gluonts.core.component import validated
from pts.model import weighted_average
from pts.modules import GaussianDiffusion, DiffusionOutput, MeanScaler, NOPScaler
from .epsilon_theta import EpsilonTheta


class TimeGradTrainingNetwork(nn.Module):
    @validated()
    def __init__(
            self,
            input_size: int,
            num_layers: int,
            num_cells: int,
            history_length: int,
            context_length: int,
            prediction_length: int,
            dropout_rate: float,
            lags_seq: List[int],
            target_dim: int,
            conditioning_length: int,
            diff_steps: int,
            loss_type: str,
            beta_end: float,
            beta_schedule: str,
            residual_layers: int,
            residual_channels: int,
            dilation_cycle_length: int,
            cardinality: List[int] = [1],
            embedding_dimension: int = 1,
            scaling: bool = True,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.target_dim = target_dim
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.history_length = history_length
        self.scaling = scaling

        assert len(set(lags_seq)) == len(lags_seq), "no duplicated lags allowed!"
        lags_seq.sort()
        self.lags_seq = lags_seq

        # Заменяем RNN на TST-энкодер
        self.tst_config = TimeSeriesTransformerConfig(
            prediction_length=prediction_length,
            context_length=context_length,
            num_time_features=input_size - target_dim,  # исключаем target dim из features
            num_static_categorical_features=1,
            num_dynamic_real_features=target_dim,
            hidden_size=num_cells,
            num_hidden_layers=num_layers,
            dropout=dropout_rate
        )
        self.tst_encoder = TimeSeriesTransformerModel(self.tst_config)

        self.denoise_fn = EpsilonTheta(
            target_dim=target_dim,
            cond_length=conditioning_length,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
            dilation_cycle_length=dilation_cycle_length,
        )

        self.diffusion = GaussianDiffusion(
            self.denoise_fn,
            input_size=target_dim,
            diff_steps=diff_steps,
            loss_type=loss_type,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
        )

        self.distr_output = DiffusionOutput(
            self.diffusion, input_size=target_dim, cond_size=conditioning_length
        )

        self.proj_dist_args = self.distr_output.get_args_proj(num_cells)

        self.embed_dim = 1
        self.embed = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.embed_dim
        )

        if self.scaling:
            self.scaler = MeanScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

    @staticmethod
    def get_lagged_subsequences(
            sequence: torch.Tensor,
            sequence_length: int,
            indices: List[int],
            subsequences_length: int = 1,
    ) -> torch.Tensor:
        """Аналогично оригинальному методу"""
        assert max(indices) + subsequences_length <= sequence_length
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...].unsqueeze(1))
        return torch.cat(lagged_values, dim=1).permute(0, 2, 3, 1)

    def unroll(
            self,
            lags: torch.Tensor,
            scale: torch.Tensor,
            time_feat: torch.Tensor,
            target_dimension_indicator: torch.Tensor,
            unroll_length: int,
            begin_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # Подготовка входных данных для TST
        lags_scaled = lags / scale.unsqueeze(-1)
        input_lags = lags_scaled.reshape((-1, unroll_length, len(self.lags_seq) * self.target_dim))

        # Эмбеддинги для индикаторов целевой переменной
        index_embeddings = self.embed(target_dimension_indicator)
        repeated_index_embeddings = (
            index_embeddings.unsqueeze(1)
                .expand(-1, unroll_length, -1, -1)
                .reshape((-1, unroll_length, self.target_dim * self.embed_dim))
        )

        # Объединяем все фичи
        inputs = torch.cat((input_lags, repeated_index_embeddings, time_feat), dim=-1)

        # Кодируем через TST
        outputs = self.tst_encoder(
            past_values=inputs,
            past_time_features=time_feat,
            past_observed_mask=torch.ones_like(inputs[:, :, 0]),  # предполагаем все наблюдения
            static_categorical_features=target_dimension_indicator.unsqueeze(1)
        ).last_hidden_state

        return outputs, outputs[:, -1:], lags_scaled, inputs  # возвращаем outputs как "состояние"

    def unroll_encoder(self, *args, **kwargs):
        """Аналогично оригинальному методу, но с новым unroll"""
        past_time_feat = kwargs['past_time_feat']
        past_target_cdf = kwargs['past_target_cdf']
        past_observed_values = kwargs['past_observed_values']
        past_is_pad = kwargs['past_is_pad']
        future_time_feat = kwargs.get('future_time_feat')
        future_target_cdf = kwargs.get('future_target_cdf')
        target_dimension_indicator = kwargs['target_dimension_indicator']

        past_observed_values = torch.min(past_observed_values, 1 - past_is_pad.unsqueeze(-1))

        if future_time_feat is None or future_target_cdf is None:
            time_feat = past_time_feat[:, -self.context_length:, ...]
            sequence = past_target_cdf
            sequence_length = self.history_length
            subsequences_length = self.context_length
        else:
            time_feat = torch.cat((past_time_feat[:, -self.context_length:, ...], future_time_feat), dim=1)
            sequence = torch.cat((past_target_cdf, future_target_cdf), dim=1)
            sequence_length = self.history_length + self.prediction_length
            subsequences_length = self.context_length + self.prediction_length

        lags = self.get_lagged_subsequences(
            sequence=sequence,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length,
        )

        _, scale = self.scaler(
            past_target_cdf[:, -self.context_length:, ...],
            past_observed_values[:, -self.context_length:, ...],
        )

        outputs, states, lags_scaled, inputs = self.unroll(
            lags=lags,
            scale=scale,
            time_feat=time_feat,
            target_dimension_indicator=target_dimension_indicator,
            unroll_length=subsequences_length,
        )

        return outputs, states, scale, lags_scaled, inputs

    # Остальные методы остаются без изменений
    def distr_args(self, rnn_outputs: torch.Tensor):
        (distr_args,) = self.proj_dist_args(rnn_outputs)
        return distr_args

    def forward(self, *args, **kwargs):
        """Аналогично оригинальному методу"""
        seq_len = self.context_length + self.prediction_length

        rnn_outputs, _, scale, _, _ = self.unroll_encoder(**kwargs)

        target = torch.cat(
            (kwargs['past_target_cdf'][:, -self.context_length:, ...],
             kwargs['future_target_cdf']),
            dim=1,
        )

        distr_args = self.distr_args(rnn_outputs=rnn_outputs)
        if self.scaling:
            self.diffusion.scale = scale

        likelihoods = self.diffusion.log_prob(target, distr_args).unsqueeze(-1)

        past_observed_values = torch.min(
            kwargs['past_observed_values'],
            1 - kwargs['past_is_pad'].unsqueeze(-1)
        )

        observed_values = torch.cat(
            (past_observed_values[:, -self.context_length:, ...],
             kwargs['future_observed_values']),
            dim=1,
        )

        loss_weights, _ = observed_values.min(dim=-1, keepdim=True)
        loss = weighted_average(likelihoods, weights=loss_weights, dim=1)

        return (loss.mean(), likelihoods, distr_args)


class TimeGradPredictionNetwork(TimeGradTrainingNetwork):
    def __init__(self, num_parallel_samples: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_parallel_samples = num_parallel_samples
        self.shifted_lags = [l - 1 for l in self.lags_seq]

    def sampling_decoder(self, *args, **kwargs):
        """Аналогично оригинальному методу, но с использованием TST states"""
        past_target_cdf = kwargs['past_target_cdf']
        target_dimension_indicator = kwargs['target_dimension_indicator']
        time_feat = kwargs['time_feat']
        scale = kwargs['scale']
        begin_states = kwargs['begin_states']

        def repeat(tensor, dim=0):
            return tensor.repeat_interleave(repeats=self.num_parallel_samples, dim=dim)

        repeated_past_target_cdf = repeat(past_target_cdf)
        repeated_time_feat = repeat(time_feat)
        repeated_scale = repeat(scale)
        if self.scaling:
            self.diffusion.scale = repeated_scale
        repeated_target_dimension_indicator = repeat(target_dimension_indicator)
        repeated_states = repeat(begin_states, dim=1)

        future_samples = []

        for k in range(self.prediction_length):
            lags = self.get_lagged_subsequences(
                sequence=repeated_past_target_cdf,
                sequence_length=self.history_length + k,
                indices=self.shifted_lags,
                subsequences_length=1,
            )

            rnn_outputs, repeated_states, _, _ = self.unroll(
                begin_state=repeated_states,
                lags=lags,
                scale=repeated_scale,
                time_feat=repeated_time_feat[:, k: k + 1, ...],
                target_dimension_indicator=repeated_target_dimension_indicator,
                unroll_length=1,
            )

            distr_args = self.distr_args(rnn_outputs=rnn_outputs)
            new_samples = self.diffusion.sample(cond=distr_args)
            future_samples.append(new_samples)
            repeated_past_target_cdf = torch.cat((repeated_past_target_cdf, new_samples), dim=1)

        samples = torch.cat(future_samples, dim=1)
        return samples.reshape((-1, self.num_parallel_samples, self.prediction_length, self.target_dim))

    def forward(self, *args, **kwargs):
        """Аналогично оригинальному методу"""
        past_observed_values = torch.min(
            kwargs['past_observed_values'],
            1 - kwargs['past_is_pad'].unsqueeze(-1)
        )

        _, begin_states, scale, _, _ = self.unroll_encoder(
            past_time_feat=kwargs['past_time_feat'],
            past_target_cdf=kwargs['past_target_cdf'],
            past_observed_values=past_observed_values,
            past_is_pad=kwargs['past_is_pad'],
            future_time_feat=None,
            future_target_cdf=None,
            target_dimension_indicator=kwargs['target_dimension_indicator'],
        )

        return self.sampling_decoder(
            past_target_cdf=kwargs['past_target_cdf'],
            target_dimension_indicator=kwargs['target_dimension_indicator'],
            time_feat=kwargs['future_time_feat'],
            scale=scale,
            begin_states=begin_states,
        )
