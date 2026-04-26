import coremltools as ct

model = ct.models.MLModel("checkpoints/2026-04-23_085355_mobilenet_v3_large/DeepScanClassifier_mobilenet_v3_large.mlpackage")

print(model.short_description)
print(model.author)
print(model.version)
print(model.license)
print(model.input_description)
print(model.output_description)
print(model.user_defined_metadata)