from transformers import BlipProcessor , BlipForConditionalGeneration,DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch

def get_image_caption(image_path):
    """
    Generates a caption for an image.
    Takes an image path as a string and returns a caption for the image.
    """
    image = Image.open(image_path).convert('RGB')
    model_name = "Salesforce/blip-image-captioning-large"  # Replace with the actual model name
    device = "cpu"
    
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

    input_data = processor(image, return_tensors="pt").to(device)
    output = model.generate(**input_data, max_new_tokens=50)#keyword arguments as **inputs

    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption


#detects object
def detect_objects(image_path): 
    image = Image.open(image_path).convert('RGB')
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    detect=""

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detect+='[ {} , {}, {}, {} ]'.format(int(box[0]),int(box[1]),int(box[2]),int(box[3]))
        detect+=' {} '.format(model.config.id2label[int(label)])
        detect+='{}\n'.format(float(score))

    return detect
        

if __name__=='__main__':
    image_path=r'D:\Chat_with_image\test_image.jpeg'
    detect=detect_objects(image_path)
    print(detect)
 




