from langchain.tools import BaseTool
from transformers import BlipProcessor , BlipForConditionalGeneration,DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch

class ImageCaptionTool(BaseTool):
    name="Image Captioner"
    description="Use this tool when given the path to the image you want to describe."\
    "it will return the simple caption describing the image "

    def _run(self,image_path):
        
        image = Image.open(image_path).convert('RGB')
        model_name = "Salesforce/blip-image-captioning-large"  # Replace with the actual model name
        device = "cpu"
        
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

        input_data = processor(image, return_tensors="pt").to(device)
        output = model.generate(**input_data, max_new_tokens=20)#keyword arguments as **inputs

        caption = processor.decode(output[0], skip_special_tokens=True)

        return caption
    
    def _arun(self,query:str):
        raise NotImplementedError("this funtion does not support async")



class ObjectdetectionTool(BaseTool):
    name="Object Detector"
    description="Use this tool when given the path to an image that you would like to detect objects."\
    "it will return a list of all detected objects .Each element in the list in the format"\
    "[x1 ,y1 ,x2, y2] class_name confidance score"

    def _run(self,image_path):
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
    
    def _arun(self,query:str):
        raise NotImplementedError("this funtion does not support async")






        