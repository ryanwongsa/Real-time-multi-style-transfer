import torch
from torchvision import transforms
from inference.Inferencer import Inferencer
from models.PasticheModel import PasticheModel
from PIL import Image

import argparse

def inference(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_styles = args.num_styles
    image_size = args.image_size
    model_save_dir = args.model_save_dir

    pastichemodel = PasticheModel(num_styles)

    inference = Inferencer(pastichemodel,device,image_size)
    inference.load_model_weights(model_save_dir)

    example_image_path = args.input_image
    im = Image.open(example_image_path).convert('RGB')
    
    style_choice = args.style_choice
    style_choice_2 = args.style_choice_2
    style_factor = args.style_factor
    
    if args.style_choice_2 == None or args.style_factor == None:
        result_img = inference.eval_image(im, style_choice)
    
    if args.style_choice_2 != None and args.style_factor != None:
        result_img = inference.eval_image(im, style_choice, style_choice_2, style_factor)
        
    result_img.save(args.output_file_name)

def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for inference on mutli-style-transfer")
    
    main_arg_parser.add_argument("--image-size", type=int, default=512,
                                  help="size of one side of the output image, default is 512")
    main_arg_parser.add_argument("--num-styles", type=int, default=16,
                                  help="number of styles in the style net used for training, default is 16")
    
    main_arg_parser.add_argument("--input-image", type=str, required=True,
                                  help="path to image to do style transfer")
    
    main_arg_parser.add_argument("--model-save-dir", type=str, required=True,
                                  help="path to file containing the model weights")
    
    main_arg_parser.add_argument("--style-choice", type=int, default=0,
                                  help="style choice, must be less than or num-styles. default is 0")
    
    main_arg_parser.add_argument("--style-choice-2", type=int, default=None,
                                  help="second style choice to mix between the two styles")
    
    main_arg_parser.add_argument("--style-factor", type=float, default=None,
                                  help="style percentage between the second a first style. must between 0-1 (inclusive)")
 
    main_arg_parser.add_argument("--output-file-name", type=str, default="output.jpg",
                                  help="name of the file to save your result to, default output.jpg")
    
    args = main_arg_parser.parse_args()
    inference(args)
    

if __name__ == '__main__':
    main()