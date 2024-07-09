import torch, os, matplotlib.pyplot as plt, numpy as np
from PIL import Image; from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from unet import UNet
from diffusionmodel import DiffusionModel




class Material:
	def __init__(self, name, max_temp, label):
		self.name = name
		self.max_temp = int(max_temp)
		self.label = torch.tensor([label])


def get_files_labeled(base_path, materials):
	selected_files=[]
	for root, dirs, files in os.walk(base_path):
		for material in materials:
			if 'Screen 1' in root:
				if material.name.lower() in root.lower():
					for file in files:
						if '.jpeg' in file:
							file_temp = int( file.split(r'.')[0] )
							if file_temp <= material.max_temp:
								selected_files.append([f"{root}\\{file}", material.label])

	return selected_files


class LabeledCustomDataset(Dataset):
	def __init__(self, files, transform):
		self.files = files
		self.transform = transform

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		filename = self.files[idx][0]
		label = self.files[idx][1]
		img = Image.open(filename).convert('RGB')
		img = img.crop((162, 26, 1074, 806)) # crop the gray area
		return self.transform(img), label













if __name__ == "__main__":

	# GLOBALS      ######
	LR = 0.0001
	epochs = 39
	batch_size = 32
	desired_min_size = 4
	needed_size = int(desired_min_size * 2**(len(UNet().sequence_channels)))
	img_shape = (needed_size, needed_size)
	###     #############


	# DATASET ###

	materials = [Material('e7', 60, 0), Material('8CB', 34, 1)]
	files = get_files_labeled(f"{os.getcwd()}\\luiz", materials)

	transform = transforms.Compose([
		transforms.ToTensor(), 
		transforms.Resize(img_shape),
		transforms.Lambda(lambda x: x*2 - 1),
		])

	reverse_transform = transforms.Compose([
	    transforms.Lambda(lambda t: (t + 1) / 2),
	    transforms.Lambda(lambda t: t.permute(1, 2, 0)),
	    transforms.Lambda(lambda t: t * 255.),
	    transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
	    transforms.ToPILImage(),
	    ])	

	dataset = LabeledCustomDataset(files, transform)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	
	print(f"Selected IMG SHAPE{img_shape} | Channels 3 | Epochs {epochs} | batch size {batch_size} | Dataset Samples: {len(dataset)}| LR {LR} | Classes ('e7', '8cb') ")

	##########  Train  ##########

	device= "cuda" if torch.cuda.is_available() else "cpu"
	net = UNet(labels=True).to(device); diffusion_model = DiffusionModel()
	loss_fn = torch.nn.MSELoss()
	opt = torch.optim.Adam(net.parameters(), lr=LR)

	for epoch in range(epochs): 
		epoch_loss = []
		for inputs, batch_labels in dataloader:
			inputs = inputs.to(device)
			opt.zero_grad()
			t = torch.randint(0, diffusion_model.timesteps, (inputs.shape[0],)).long().to(device)
			batch_noisy, noise = diffusion_model.forward(inputs, t, device)
			noise = noise.to(device)       
			pred_noise = net(batch_noisy, t, labels = batch_labels.reshape(-1,1).float().to(device))
			loss = loss_fn(pred_noise, noise)
			loss.backward()
			opt.step()
			epoch_loss.append(loss.item())
		print(f"EPOCH {epoch+1} LOSS {np.mean(epoch_loss)}")



	classes = ('e7', '8cb')
	NUM_CLASSES = len(classes)
	NUM_DISPLAY_IMAGES = 4
	f, ax = plt.subplots(NUM_CLASSES, NUM_DISPLAY_IMAGES, figsize = (20,20))

	with torch.no_grad():
		for c in range(NUM_CLASSES):
			imgs = torch.randn((NUM_DISPLAY_IMAGES, 3) + img_shape).to(device)

			for i in reversed(range(diffusion_model.timesteps)):
				t = torch.full((1,), i, dtype=torch.long, device=device)
				labels = torch.tensor([c]).float().to(device)
				imgs = diffusion_model.backward(imgs, t, net.eval(), labels=labels)

			for idx, img in enumerate(imgs):
				ax[c][idx].imshow(reverse_transform(img))
				ax[c][idx].set_title(f"Class: {classes[c]}", fontsize = 10)

	plt.show()