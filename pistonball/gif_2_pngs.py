from PIL import Image
import sys

def gif_to_pngs(gif_file, output_base_name):
    try:
        image = Image.open(gif_file)
    except IOError:
        print(f"Error: Unable to open file {gif_file}")
        return

    frame_number = 0
    while True:
        output_filename = f"{output_base_name}_{frame_number:03d}.png"
        try:
            image.save(output_filename)
            print(f"Saved frame {frame_number} as {output_filename}")
        except IOError:
            print(f"Error: Unable to save file {output_filename}")
            break

        frame_number += 1
        try:
            image.seek(image.tell() + 1)
        except EOFError:
            break

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python gif_to_pngs.py <input_gif> <output_base_name>")
    else:
        gif_to_pngs(sys.argv[1], sys.argv[2])
