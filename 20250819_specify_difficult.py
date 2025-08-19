import os
import shutil

# source directory containing your XML files
xml_dir = "/annotations"
# target directory for modified copies
output_dir = os.path.join(xml_dir, "difficult")

# create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(xml_dir):
    if file.endswith(".xml"):
        src_path = os.path.join(xml_dir, file)
        dst_path = os.path.join(output_dir, file)

        # copy the original file first
        shutil.copy2(src_path, dst_path)

        with open(dst_path, "r", encoding="utf-8") as f:
            content = f.read()

        if "<difficult>0</difficult>" in content:
            new_content = content.replace("<difficult>0</difficult>", "<difficult>1</difficult>")
            with open(dst_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"Copied & updated: {file}")
        else:
            print(f"Copied (no change): {file}")

print(f"All files processed. Modified copies are in: {output_dir}")
