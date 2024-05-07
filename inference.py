import requests
from io import BytesIO
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import os, asyncio

n_classes = 120

label_to_name = {0: 'Chihuahua', 1: 'Japanese spaniel', 2: 'Maltese dog', 3: 'Pekinese', 4: 'Shih Tzu',
                 5: 'Blenheim spaniel', 6: 'papillon', 7: 'toy terrier', 8: 'Rhodesian ridgeback',
                 9: 'Afghan hound', 10: 'basset', 11: 'beagle', 12: 'bloodhound', 13: 'bluetick',
                 14: 'black and tan coonhound', 15: 'Walker hound', 16: 'English foxhound', 17: 'redbone',
                 18: 'borzoi', 19: 'Irish wolfhound', 20: 'Italian greyhound', 21: 'whippet',
                 22: 'Ibizan hound', 23: 'Norwegian elkhound', 24: 'otterhound', 25: 'Saluki',
                 26: 'Scottish deerhound', 27: 'Weimaraner', 28: 'Staffordshire bullterrier',
                 29: 'American Staffordshire terrier', 30: 'Bedlington terrier', 31: 'Border terrier',
                 32: 'Kerry blue terrier', 33: 'Irish terrier', 34: 'Norfolk terrier', 35: 'Norwich terrier',
                 36: 'Yorkshire terrier', 37: 'wire haired fox terrier', 38: 'Lakeland terrier',
                 39: 'Sealyham terrier', 40: 'Airedale', 41: 'cairn', 42: 'Australian terrier', 43: 'Dandie Dinmont',
                 44: 'Boston bull', 45: 'miniature schnauzer', 46: 'giant schnauzer', 47: 'standard schnauzer',
                 48: 'Scotch terrier', 49: 'Tibetan terrier', 50: 'silky terrier', 51: 'soft coated wheaten terrier',
                 52: 'West Highland white terrier', 53: 'Lhasa', 54: 'flat coated retriever', 55: 'curly coated retriever',
                 56: 'golden retriever', 57: 'Labrador retriever', 58: 'Chesapeake Bay retriever',
                 59: 'German short haired pointer', 60: 'vizsla', 61: 'English setter', 62: 'Irish setter',
                 63: 'Gordon setter', 64: 'Brittany spaniel', 65: 'clumber', 66: 'English springer', 67: 'Welsh springer spaniel',
                 68: 'cocker spaniel', 69: 'Sussex spaniel', 70: 'Irish water spaniel', 71: 'kuvasz', 72: 'schipperke',
                 73: 'groenendael', 74: 'malinois', 75: 'briard', 76: 'kelpie', 77: 'komondor', 78: 'Old English sheepdog',
                 79: 'Shetland sheepdog', 80: 'collie', 81: 'Border collie', 82: 'Bouvier des Flandres', 83: 'Rottweiler',
                 84: 'German shepherd', 85: 'Doberman', 86: 'miniature pinscher', 87: 'Greater Swiss Mountain dog',
                 88: 'Bernese mountain dog', 89: 'Appenzeller', 90: 'EntleBucher', 91: 'boxer', 92: 'bull mastiff',
                 93: 'Tibetan mastiff', 94: 'French bulldog', 95: 'Great Dane', 96: 'Saint Bernard', 97: 'Eskimo dog',
                 98: 'malamute', 99: 'Siberian husky', 100: 'affenpinscher', 101: 'basenji', 102: 'pug', 103: 'Leonberg',
                 104: 'Newfoundland', 105: 'Great Pyrenees', 106: 'Samoyed', 107: 'Pomeranian', 108: 'chow', 109: 'keeshond',
                 110: 'Brabancon griffon', 111: 'Pembroke', 112: 'Cardigan', 113: 'toy poodle', 114: 'miniature poodle',
                 115: 'standard poodle', 116: 'Mexican hairless', 117: 'dingo', 118: 'dhole', 119: 'African hunting dog'}

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict_image(model, image):
    model.eval()
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class, probabilities.squeeze().tolist()

def load_image(image_path):
    if isinstance(image_path, str) and image_path.startswith('http'):
        # Load image from URL
        print("loading from url")
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content))
    else:
        print("loading from file path: ", image_path)
        # Load image from local file path
        image = Image.open(image_path)
    return image

def inference(model_path, image_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained model
    model = resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, n_classes)  # Modify the last layer
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("model loaded...")

    # results
    predicted_labels = []
    images = []

    # Load and plot the original image
    print("loading image...")
    image = load_image(image_path)
    print("loaded image...")
    images.append(image)
    print(images)
    image = preprocess_image(image).to(device)
    print(images)
    # Predict
    predicted_class, probabilities = predict_image(model, image)
    print("1. ", predicted_class)
    predicted_name = label_to_name[predicted_class]
    print("2. ", predicted_name)
    predicted_probability = probabilities[predicted_class] * 100
    print("3. ", predicted_probability)
    # Store results and Print to stdout
    predicted_labels.append(predicted_name)
    print(f"Predicted breed: {predicted_name} [{predicted_probability:.2f}%]")

    return predicted_labels, images

async def predict_breed(image_url=None):
    loop = asyncio.get_event_loop()
    print("predicting: ", image_url)
    return await loop.run_in_executor(None, pred, image_url)

def pred(image_url=None):
    return inference(model_path='fine_tuned_resnet50.pth', image_path=image_url)
