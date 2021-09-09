from ..convmlp import convmlp_s, convmlp_m, convmlp_l
from timm.models.registry import register_model


@register_model
def convmlp_s_classification(*args, **kwargs):
    return convmlp_s(classifier_head=True, *args, **kwargs)


@register_model
def convmlp_m_classification(*args, **kwargs):
    return convmlp_m(classifier_head=True, *args, **kwargs)


@register_model
def convmlp_l_classification(*args, **kwargs):
    return convmlp_l(classifier_head=True, *args, **kwargs)
