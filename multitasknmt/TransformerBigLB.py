import opennmt

class TransformerBigLB(opennmt.models.Transformer):
    def __init__(self):
        super().__init__(
            num_layers=6,
            num_units=1024,
            num_heads=16,
            ffn_inner_dim=4096,
            share_embeddings=opennmt.models.EmbeddingsSharingLevel.AUTO,
        )

    def auto_config(self, *args, **kwargs):
        config = super().auto_config(*args, **kwargs)
        return opennmt.merge_config(
            config,
            {
                "data": {
                    "source_sequence_controls": {"end": True},
                },
                "params": {
                    "optimizer": "Adam",
                    "optimizer_params": {
                        "beta_1": 0.9,
                        "beta_2": 0.98,
                        "epsilon": 1e-8,
                    },
                    "learning_rate": 0.001,
                    "decay_type": "InvSqrtDecay",
                    "decay_params": {
                        "warmup_steps": 4000,
                        "initial_learning_rate": 1e-7,
                    },
                },
                "train": {
                    "batch_size": 0,
                    "effective_batch_size": 400000,
                    "save_summary_steps": 10,
                },
            },
        )


model = TransformerBigLB
