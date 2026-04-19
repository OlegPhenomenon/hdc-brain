import random
import utility

emojis_hdv = {
        '💦': utility.hdv(),
        '🌅': utility.hdv(),
        '🥚': utility.hdv(),
        '🐺': utility.hdv(),
        '🚒': utility.hdv(),
        '🌱': utility.hdv(),
        '🍌': utility.hdv()
        }

def banana_color():
    return {
        "r": random.uniform(0.85, 1.0),
        "g": random.uniform(0.75, 0.95),
        "b": random.uniform(0.0, 0.2),
    }

def water_color():
    return {
        "r": random.uniform(0.0, 0.2),
        "g": random.uniform(0.3, 0.6),
        "b": random.uniform(0.7, 1.0),
    }

def egg_color():
    return {
        "r": random.uniform(0.95, 1.0),
        "g": random.uniform(0.9, 1.0),
        "b": random.uniform(0.7, 0.9),
    }

def sunset_color():
    return {
        "r": random.uniform(0.8, 1.0),
        "g": random.uniform(0.3, 0.7),
        "b": random.uniform(0.0, 0.3),
    }

def wolf_color():
    return {
        "r": random.uniform(0.3, 0.6),
        "g": random.uniform(0.3, 0.6),
        "b": random.uniform(0.3, 0.6),
    }

def plant_color():
    return {
        "r": random.uniform(0.0, 0.3),
        "g": random.uniform(0.5, 1.0),
        "b": random.uniform(0.0, 0.3),
    }

def firetruck_color():
    return {
        "r": random.uniform(0.8, 1.0),
        "g": random.uniform(0.0, 0.2),
        "b": random.uniform(0.0, 0.2),
    }
    
toy_data1 = [
    ("🍌", banana_color()),
    ("💦", water_color()),
    ("🥚", egg_color()),
    ("🌅", sunset_color()),
    ("🌅", sunset_color()),
    ("🌅", sunset_color()),
    ("🥚", egg_color()),
    ("🐺", wolf_color()),
    ("🐺", wolf_color()),
    ("🐺", wolf_color()),
    ("🌱", plant_color()),
    ("🌱", plant_color()),
    ("🚒", firetruck_color()),
    ("🥚", egg_color()),
    ("💦", water_color()),
    ("🌱", plant_color()),
    ("🚒", firetruck_color()),
    ("💦", water_color()),
    ("💦", water_color()),
    ("🍌", banana_color()),
]

steps = [round(i * 0.05, 2) for i in range (21)]


def generate_continues_embeddings_for_hdvs(steps):
  hdvs = [utility.hdv() for _ in range(len(steps))]
  
  for i in range(1, len(hdvs)):
    
    for j in range(len(hdvs[0])):
      
      if random.random() < 1 / len(hdvs):
        hdvs[i][j] = -hdvs[i-1][j]
      else:
        hdvs[i][j] = hdvs[i-1][j]
    
  return hdvs


def steps_to_index(step):
  return round(step * 20)

reds_hdv = generate_continues_embeddings_for_hdvs(steps)
blues_hdv = generate_continues_embeddings_for_hdvs(steps)
greens_hdv = generate_continues_embeddings_for_hdvs(steps)

def encode_color(red, green, blue):
  red_hdv = reds_hdv[steps_to_index(red)]
  green_hdv = greens_hdv[steps_to_index(green)]
  blue_hdv = blues_hdv[steps_to_index(blue)]
  
  # превратим в один цвет все три оттенка
  return utility.bind_binary(utility.bind_binary(red_hdv, green_hdv), blue_hdv)

def bind_emoji_to_color(emojis):
  schema = []
  for emoji in emojis:
    e, c = emoji
    schema.append(utility.bind_binary(emojis_hdv[e], encode_color(c['r'], c['g'], c['b'])))
    
  return schema

def random_color():
  return {
    'red': random.random(),
    'green': random.random(),
    'blue': random.random()
  }

coloried_emojis = bind_emoji_to_color(toy_data1)
memory = utility.bundle_binary(coloried_emojis)
bank_of_colors = [
  {"color": color, "hdv": encode_color(color["red"], color["green"], color["blue"])}
  for color in [random_color() for _ in range(1000)]
]


def find_color_of_emoji(emoji):
  hdv_of_emoji = emojis_hdv[emoji]
  hdv_color = utility.bind_binary(hdv_of_emoji, memory)
  
  best_color = None
  best_score = -float("inf")
  
  for item in bank_of_colors:
    score = utility.cos_similarity_binary(hdv_color, item["hdv"])
    
    if score > best_score:
      best_score = score
      best_color = item["color"]
      
  return best_color

def find_emoji_of_color(color):
  hdv_of_color = encode_color(color["red"], color["green"], color["blue"])
  hdv_emoji = utility.bind_binary(memory, hdv_of_color)
  
  best_emoji = None
  best_score = -float("inf")
  
  for emoji in emojis_hdv:
    score = utility.cos_similarity_binary(hdv_emoji, emojis_hdv[emoji])
    
    if score > best_score:
      best_score = score
      best_emoji = emoji
      
  return best_emoji

    
def print_color_block(color):
    r = int(color["red"] * 255)
    g = int(color["green"] * 255)
    b = int(color["blue"] * 255)

    print(f"\033[48;2;{r};{g};{b}m    \033[0m", color)

color = find_color_of_emoji("🥚")

print('--- 1')
print(print_color_block(color))
print('--- 1')

color = {
    "red": random.uniform(0.3, 0.6),
    "green": random.uniform(0.3, 0.6),
    "blue": random.uniform(0.3, 0.6),
}

emoji = find_emoji_of_color(color)
    
print('--- 2')
print(emoji)
print('--- 2')