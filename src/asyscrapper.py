import asyncio
import aiohttp
import csv
import os
import ssl

# Fix SSL pour Mac/Windows
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Nouvelle API de secours (TheMealDB) - Rapide et ultra fiable
API_URL_TEMPLATE = "https://www.themealdb.com/api/json/v1/1/filter.php?i={}"

CATEGORIES = ["milk", "bread"]

async def fetch_category(session, category):
    print(f"-> Recherche de nourriture pour la categorie '{category}'...")
    url = API_URL_TEMPLATE.format(category)
    try:
        async with session.get(url, ssl=ssl_context) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("meals") or []
            else:
                print(f"Erreur API : {resp.status}")
    except Exception as e:
        print(f"Erreur de connexion : {e}")
    return []

async def download_image(session, url, image_id, category):
    if not url: 
        return None
        
    folder = f"data/raw/images/{category}"
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"{image_id}.jpg")
    
    if os.path.exists(filename): 
        return filename

    try:
        async with session.get(url, ssl=ssl_context) as resp:
            if resp.status == 200:
                content = await resp.read()
                with open(filename, "wb") as f:
                    f.write(content)
                return filename
    except Exception:
        pass
    return None

def save_to_csv(filename, rows):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["foodId", "label", "category", "image"])
        writer.writerows(rows)

async def main():
    async with aiohttp.ClientSession() as session:
        for category in CATEGORIES:
            meals = await fetch_category(session, category)
            valid_rows = []
            
            print(f"-> Telechargement de {len(meals)} images reelles pour '{category}'...")
            
            tasks = []
            for meal in meals:
                meal_id = meal.get("idMeal")
                image_url = meal.get("strMealThumb")
                
                if meal_id and image_url:
                    tasks.append(download_image(session, image_url, meal_id, category))
            
            results = await asyncio.gather(*tasks)
            
            for meal, downloaded_path in zip([m for m in meals if m.get("strMealThumb")], results):
                if downloaded_path:
                    valid_rows.append([
                        meal.get("idMeal"),
                        meal.get("strMeal"),
                        category,
                        f"{meal.get('idMeal')}.jpg"
                    ])
            
            if valid_rows:
                output_file = f"data/raw/metadata_{category}_{len(valid_rows)}.csv"
                save_to_csv(output_file, valid_rows)
                print(f"SUCCES : {len(valid_rows)} vraies images sauvegardees pour {category} !\n")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
    