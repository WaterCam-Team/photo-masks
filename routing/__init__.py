"""The labeling-route RL agent: environment, inner SegFormer loop, policies.

The agent picks a route for each scene in a batch, the routed masks join the
labeled set, SegFormer is fine-tuned on it and scored on a fixed validation
set, and that score and the annotator time the routes cost make the reward.
`finetune.py` is that inner fine-tune-and-score step.
"""
