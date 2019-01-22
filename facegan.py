from aitk.templates.templates import IsaacGAN

input_data_filepath = "/tmp/faces/lfw-deepfunneled"

generator_output_filepath = "/model_output"

gan = IsaacGAN(input_data_filepath, generator_output_filepath, checkpoint=25, epochs=2000, batchsize=256)

gan.train_isaac_gan()
