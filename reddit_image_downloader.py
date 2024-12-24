import os
import requests
import time

def download_images(subreddit, output_folder):
    """
    Download as many images as possible from a subreddit.

    :param subreddit: Name of the subreddit (e.g., 'pics')
    :param output_folder: Folder to save the downloaded images
    """
    base_url = f"https://www.reddit.com/r/{subreddit}/top.json"
    headers = {"User-Agent": "reddit-image-downloader"}

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    downloaded = 0
    after = None

    while True:
        params = {"limit": 100}
        if after:
            params["after"] = after

        try:
            response = requests.get(base_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            posts = data.get("data", {}).get("children", [])

            if not posts:
                print(f"No more posts to fetch for r/{subreddit}.")
                break

            for post in posts:
                image_url = post["data"].get("url")
                if image_url and any(image_url.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif"]):
                    try:
                        image_name = f"{subreddit}_{downloaded + 1}{os.path.splitext(image_url)[-1]}"
                        image_path = os.path.join(output_folder, image_name)

                        if not os.path.exists(image_path):
                            print(f"Downloading {image_url}...")
                            img_response = requests.get(image_url, stream=True)
                            img_response.raise_for_status()

                            with open(image_path, "wb") as f:
                                for chunk in img_response.iter_content(chunk_size=8192):
                                    f.write(chunk)

                            downloaded += 1

                    except Exception as e:
                        print(f"Error downloading {image_url}: {e}")

            after = data.get("data", {}).get("after")
            if not after:
                print(f"No more posts to fetch for r/{subreddit}.")
                break

            time.sleep(1)  # Avoid hitting the Reddit API rate limit

        except requests.RequestException as e:
            print(f"Request failed for r/{subreddit}: {e}")
            break

    print(f"Downloaded {downloaded} images from r/{subreddit} to {output_folder}.")

if __name__ == "__main__":
    top_subreddits = [
        "pics", "aww", "wallpapers", "EarthPorn", "Art", "funny", "gaming", "memes", "gifs", "food",
        "history", "sports", "movies", "space", "cats", "dogs", "travel", "photography", "nature", "design",
        "technology", "anime", "music", "books", "cars", "architecture", "fashion", "drawing", "quotes", "DIY",
        "science", "comics", "lego", "coffee", "gardening", "skateboarding", "hiking", "fitness", "beer", "crafts",
        "tattoos", "boardgames", "cosplay", "programming", "dataisbeautiful", "digital_art", "streetart", "historyporn",
        "machinelearning", "spaceporn", "earthporn", "militaryporn", "mechporn", "architectureporn", "mapPorn",
        "cityporn", "winterporn", "waterporn", "wallpaperporn", "exposureporn", "adrenalineporn", "animalporn",
        "autumnporn", "aviationporn", "beachporn", "botanicalporn", "carporn", "castleporn", "destructionporn",
        "geologyporn", "gunporn", "iceporn", "industrialporn", "infrastructureporn", "japanporn", "lakeporn",
        "mapporncirclejerk", "metalporn", "museumporn", "pathporn", "powerwashingporn", "propagandaposters",
        "puzzleporn", "roadporn", "ruralporn", "seaporn", "skyporn", "stairporn", "toolporn", "urbexporn", "waterporn",
        "winterporn", "woodworkingporn"
    ]

    output_dir = input("Enter the output folder: ").strip()

    for subreddit_name in top_subreddits:
        print(f"Downloading images from r/{subreddit_name}...")
        download_images(subreddit=subreddit_name, output_folder=output_dir)
        print(f"Pausing for 10 seconds to avoid rate limiting...")
        time.sleep(10)
