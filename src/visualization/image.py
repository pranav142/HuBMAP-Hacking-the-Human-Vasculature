import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
import pandas as pd

def create_empty_masks(IMAGE_WIDTH:int=512, IMAGE_HEIGHT:int=512, NUM_CHANNELS:int=3):
    assert IMAGE_WIDTH > 0 and IMAGE_HEIGHT > 0 and NUM_CHANNELS > 0, "Please Enter Valid image stats"
    glomerulus_mask = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS), dtype=np.uint8)
    blood_vessel_mask = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS), dtype=np.uint8)
    unsure_mask = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS), dtype=np.uint8)
    return  {'glomerulus': glomerulus_mask,'blood_vessel':blood_vessel_mask ,'unsure':unsure_mask}

def fill_masks_with_annotations(image_id: str, masks: dict[str, np.array], polygons_df: pd.DataFrame):
    try: 
        annots = polygons_df.loc[polygons_df["id"] == image_id, 'annotations'].iloc[-1]
    except error as e: 
        print("Could not find image please make sure you entered the correct id and that the image has a corresponding annotation")
        return masks
    
    for annot in annots:
        annot_type = annot['type']
        coordinates = annot['coordinates']
        color = np.random.randint(0, 255, size=3).tolist()  # Generate a unique color for each instance
        cv2.fillPoly(masks[annot_type], [np.array(coordinates)], color)

    return masks

def plot_masks(image_id: str) -> None:
    
    empty_masks = create_empty_masks()
    masks = fill_masks_with_annotations(image_id, empty_masks)
    
    fig, axs = plt.subplots(1, 3, figsize=(8,8))
    axs[0].imshow(masks['glomerulus'], cmap='gray')
    axs[0].set_title('glomerulus mask')

    axs[1].imshow(masks['blood_vessel'], cmap='gray')
    axs[1].set_title('blood vessel mask')

    axs[2].imshow(masks['unsure'], cmap='gray')
    axs[2].set_title('unsure mask')
    
    plt.tight_layout()
    plt.show()

def combine_masks(masks: dict[str, np.array]):
    glomerulus_mask  = masks['glomerulus']
    blood_vessel_mask = masks['blood_vessel']
    unsure_mask = masks['unsure']

    combined_image = np.concatenate([glomerulus_mask, blood_vessel_mask, unsure_mask], axis=1)
    
    return combined_image

def get_image(image_id: str, base_dir: str) -> np.array:
    image_path = base_dir + "train/" + image_id + ".tif"
    image = imageio.v2.imread(image_path)
    return image

def plot_image(image_id: str, base_dir: str) -> None:
    image = get_image(image_id, base_dir)
    plt.imshow(image)
    plt.title(f"Image Id: {image_id.split()[0]}, Image Shape: {image.shape}")
    plt.show()

def plot_masks_and_original_image(image_id: str, polygons_df: pd.DataFrame, base_dir: str) -> None:
    if image_id.endswith('.tif'):
        image_id = image_id.split()[0]
    empty_masks = create_empty_masks()
    masks = fill_masks_with_annotations(image_id, empty_masks, polygons_df)
    image = get_image(image_id, base_dir)
    mask_overlay_image, _ = get_mask_image_overlay(image_id, polygons_df, base_dir)
    
    fig, axs = plt.subplots(1, 5, figsize=(16,16))
    axs[0].imshow(masks['glomerulus'], cmap='gray')
    axs[0].set_title('glomerulus mask')

    axs[1].imshow(masks['blood_vessel'], cmap='gray')
    axs[1].set_title('blood vessel mask')

    axs[2].imshow(masks['unsure'], cmap='gray')
    axs[2].set_title('unsure mask')
    
    axs[3].imshow(image)
    axs[3].set_title('Ground Truth')
    
    axs[4].imshow(mask_overlay_image)
    axs[4].set_title("mask on top of image")
    
    for ax in axs.flat:
        ax.set_xticks([]) 
        ax.set_yticks([]) 

    plt.subplots_adjust(top=0.9)

    plt.suptitle(f"Masks and Original Image Example: {image_id}", fontsize=16, y=0.63) 
    plt.tight_layout()
    plt.show()

def get_mask_image_overlay(image_id: str, polygons_df: pd.DataFrame, base_dir: str) -> tuple[np.array, dict[str, int]]:  
    annots = polygons_df.loc[polygons_df["id"] == image_id, 'annotations'].iloc[-1]
    img = get_image(image_id, base_dir)
    
    RED= (255,0,0)
    GREEN= (0,255,0)
    BLUE= (0,0,255)
    color_map = {'blood_vessel':RED, 'glomerulus':BLUE, 'unsure':GREEN}
    instance_count = {'blood_vessel':0, 'glomerulus':0, 'unsure':0}
    
    for annot in annots:
        color = color_map[annot['type']]
        instance_count[annot['type']]+=1
        coords = np.array(annot['coordinates'])
        cv2.polylines(img, coords, True, color, 3)
    
    return img, instance_count

def draw_mask_image_overlay(image_id: str, polygons_df: pd.DataFrame, base_dir:str) -> None:
    img, instance_count = get_mask_image_overlay(image_id, polygons_df, base_dir)
    
    fig, axs = plt.subplots(figsize=(10,10))
    
    print(f"Blood Vessels (Red) Count: {instance_count['blood_vessel']}")
    print(f"Glomerulus (Blue) Count: {instance_count['glomerulus']}")
    print(f"Unsure (Green) Count: {instance_count['unsure']}")

    axs.imshow(img)
    plt.title(f"Image Id: {image_id}")
    plt.axis('off')
    plt.show()