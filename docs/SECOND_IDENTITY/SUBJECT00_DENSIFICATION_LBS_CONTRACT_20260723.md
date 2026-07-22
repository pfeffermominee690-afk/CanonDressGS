# Subject00 Densification LBS Contract (2026-07-23)

## Scope

The audited runtime has no clone, split, densification, or prune implementation. This task therefore simulated metadata behavior offline for 10,000 deterministic parent Gaussians. It did not train, alter runtime code, or create a training checkpoint.

## Frozen operations

- **Clone:** inherit face ID, barycentric coordinates, component, region, and 55 weights bitwise.
- **Split small:** retain the parent face and reproject within that face under the frozen local rule; recompute weights from the resulting barycentric coordinates.
- **Split large:** invoke the deterministic global closest-triangle fallback and record all component, region, dominant-joint, and tie changes.
- **Prune:** apply one identical index selection to geometry, weights, and every attachment tensor.
- **Checkpoint-format smoke:** save and restore prototype arrays only; it is not a training checkpoint.

## Results

Clone inheritance was exact for every attachment field and all weights. Small split retained the parent face for 10,000/10,000 samples, had zero component and region switches, an interpolation-formula maximum error of 0, six dominant-joint switches, mean/max canonical displacement of `0.0002841332541808389`/`0.0009763457658498644` metres, and mean/max parent-child weight drift of `3.0059645047324828e-05`/`0.010769248008728027`.

Large split sent all 10,000 samples to the fallback. It recorded 35 component switches, 459 region switches, 1,106 dominant-joint switches, and 1,554 ties. Mean/max distance to the selected surface was `0.014077906048488796`/`0.029993699804179787` metres. These failures are not hidden: they agree with the independent narrow-band finding that unrestricted global fallback is insufficient.

Pruning retained 5,000/10,000 records with exact metadata index parity. Prototype checkpoint arrays matched after roundtrip, and independent runs produced byte-identical checkpoint files with SHA256 `559c57da8bdd3ebcccc61bb818503d3c43a931f2a8352976b5168dbf683a5154`.

At frame 0, clone deformation matched its parent exactly. Small-split parent-child deformation distance was mean `0.00028435501735657454`, max `0.0010508766863495111`; large-split distance was mean `0.029989056289196014`, max `0.04275168851017952`. All deformation outputs were finite.

## Contract disposition

The metadata/inheritance/checkpoint mechanics pass their offline gate. Large-split scientific suitability does not pass merely because the operation is deterministic: before production use, its fallback must be region/component gated. A future runtime implementation must update or prune geometry and every attachment field atomically and must never silently drop provenance.
