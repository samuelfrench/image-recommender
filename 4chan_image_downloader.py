import os
import requests
from bs4 import BeautifulSoup

def download_images_from_thread(thread_url, download_folder):
    try:
        # Fetch the thread page
        response = requests.get(thread_url)
        response.raise_for_status()

        # Parse the page with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all image elements
        images = soup.find_all('a', class_='fileThumb')

        if not images:
            print(f"No images found on the thread: {thread_url}")
            return

        # Create the download folder if it doesn't exist
        os.makedirs(download_folder, exist_ok=True)

        # Download each image
        for img in images:
            img_url = "https:" + img['href']
            img_name = img_url.split('/')[-1]
            img_path = os.path.join(download_folder, img_name)

            print(f"Downloading {img_name} from {thread_url}...")
            with requests.get(img_url, stream=True) as img_response:
                img_response.raise_for_status()
                with open(img_path, 'wb') as file:
                    for chunk in img_response.iter_content(chunk_size=8192):
                        file.write(chunk)

        print(f"Downloaded {len(images)} images from {thread_url} to {download_folder}.")

    except requests.RequestException as e:
        print(f"An error occurred while processing {thread_url}: {e}")

def download_images_from_board(board_url, download_folder):
    try:
        # Fetch the board page
        response = requests.get(board_url)
        response.raise_for_status()

        # Parse the page with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all thread links
        threads = soup.find_all('a', href=True)
        thread_urls = ["https://boards.4channel.org" + thread['href'] for thread in threads if '/thread/' in thread['href']]

        if not thread_urls:
            print("No threads found on the board.")
            return

        # Download images from each thread
        for thread_url in thread_urls:
            download_images_from_thread(thread_url, download_folder)

    except requests.RequestException as e:
        print(f"An error occurred while processing the board: {e}")

if __name__ == "__main__":
    board_url = input("Enter the URL of the 4chan board (e.g., https://boards.4channel.org/g/): ").strip()
    download_folder = input("Enter the folder where images will be saved: ").strip()
    download_images_from_board(board_url, download_folder)

