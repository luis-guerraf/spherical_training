Training networks on a sphere (full precision and binary)

Usage: python main.py --project --retract

Remarks: 
- DO NOT use weight-decay, momentum or ADAM as they will push the gradient away from the tangent hyperplane (keep an eye on the unit norm of the layers)
- Two retraction methods available in the retraction function: Normalization and Exponential mapping

To do: 
- Finish extension to binary networks
- Extend notion of momentum (spherical momentum)?
- Extend to Stiefel manifold for orthogonal channels