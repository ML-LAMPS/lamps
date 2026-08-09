PER_MODEL_GP_LENGTH_SCALE ?= 50.0
PER_MODEL_GP_SIGNAL_STD ?= 1.0
PER_MODEL_GP_NOISE_STD ?= 0.05
PER_MODEL_GP_JITTER ?= 1e-8

.PHONY: cache

cache: \
	.cache/per_model_log_loss_gp/image-classification \
	.cache/per_model_log_loss_gp/text-classification \
	.cache/per_model_log_loss_gp/machine-translation

.cache/per_model_log_loss_gp/%:
	python precompute_independent_loss.py \
		--experiment "$*" \
		--datasets all \
		--length-scale "$(PER_MODEL_GP_LENGTH_SCALE)" \
		--signal-std "$(PER_MODEL_GP_SIGNAL_STD)" \
		--noise-std "$(PER_MODEL_GP_NOISE_STD)" \
		--jitter "$(PER_MODEL_GP_JITTER)"
