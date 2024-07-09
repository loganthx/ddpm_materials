# ddpm_materials
Aplication of a denoising diffusion algorithm, with the goal of generating image samples of materials

# Materials
materials = ('7e', '8cb')

# Small Dataset
We have a small dataset, containing around 600 imagesclassified by the name of the material and it's temperature. Within those images, we have little variety of pixel distributios, because an image of the material makes very small changes along the temperature increasing process, except for the limit temperature, which suddenly changes the pixel distribution of the sample significantly. Knowing that, we will test our UNet, derived from dtransposed and changed so it better suits our purposes.

# TEST 1: Training with 2 Classes:
We trained our model with the images and material names (not informing the temperatures to the model, and not considering t > t_max=limit). These are the generated images:

![](https://i.ibb.co/JdGkSRf/Figure-2.png)




# TEST 2: Our UNet ability to generate samples with semantic significancy
It's reasonable to doubt the previous results as a prove of the network good performance, because we feed many images that look alike, our dataset ends up being poor and the network might overfit the data. To test our net capacity of generating different samples, we train it on a completely different dataset (considering only one class of image), and see the results.
