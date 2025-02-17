import os
import requests
from tqdm import tqdm

def download_file(url, save_folder):
    """Download a file from a given URL and save it in the specified folder."""
    filename = os.path.join(save_folder, os.path.basename(url))  # Extract file name
    try:
        response = requests.get(url, stream=True)  # Stream download for large files
        response.raise_for_status()  # Raise error if download fails
    except Exception as e:
        print(f"following error happen on url <{url}>: ", e)
        return
    
    with open(filename, "wb") as file:
        for chunk in response.iter_content(1024):  # Download in chunks
            file.write(chunk)


def main():
    # Load list of rasters to download
    with open('./src/proj-vegroof-example-rasters.txt', 'r') as infile:
        lst_rasters = infile.read()
    lst_rasters = lst_rasters.split('\n')

    # Create architecture
    if not os.path.exists('./data/sources/example/rasters'):
        os.makedirs('./data/sources/example/rasters')
    if not os.path.exists('./data/sources/example/AOI'):
        os.makedirs('./data/sources/example/AOI')

    # Load rasters
    print("Loading rasters ..")
    for _, file_url in tqdm(enumerate(lst_rasters), total=len(lst_rasters), desc="Processing"):
        download_file(file_url, "./data/sources/example/rasters")
    print("Done!")

    # Load AOI
    print("Loading AOI")
    download_file("https://data.stdl.ch/proj-vegroof/example/aoi/roofs_example.gpkg", "./data/sources/example/AOI")
    print("Done!")


if __name__ == '__main__':
    main()