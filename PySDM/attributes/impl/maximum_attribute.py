"""
logic around `PySDM.attributes.impl.maximum_attribute.MaximumAttribute` - parent class
 for attributes for which under coalescence the newly collided particle's attribute
 value is set to maximum of values of colliding particle (e.g., freezing temperature
  in singular immersion freezing)
"""

from .base_attribute import BaseAttribute


class MaximumAttribute(BaseAttribute):
    pass
