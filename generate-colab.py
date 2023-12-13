from pipelines import SimplePipeline
from PIL import Image

pipeline = SimplePipeline()

checkpoint_dir = "checkpoints/pipeline-test.pt"

pipeline.load_from_file(checkpoint_dir)

imgs: Image = pipeline.generate()

imgs.save("output.png")
