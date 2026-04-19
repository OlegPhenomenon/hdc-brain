import random

N = 10000

def hdv(n = N):
    return [random.choice((-1,1)) for _ in range(n)]


def bundle(hdv_matrix):
    size = len(hdv_matrix[0])
    hdv_matrix_size = len(hdv_matrix)
    result = []

    for i in range(size):
        total = 0

        for n in range(hdv_matrix_size):
            total = total + hdv_matrix[n][i] 

        if total > 0:
            result.append(1)
        elif total < 0:
            result.append(-1)
        else:
            result.append(1)

    return result

def bind(a, b):
    result = []

    for i in range(len(a)):
        result.append(a[i] * b[i])

    return result

def shift(v, k=1):
    k = k % len(v)
    return v[-k:] + v[:-k]

def shift_left(v, k=1):
    k = k % len(v)
    return v[k:] + v[:k]


def cos_similarity(a, b):
    if len(a) != len(b):
        raise ValueError("Vectors should be the same")

    dot_product = 0
    norm_a_sq = 0
    norm_b_sq = 0

    for x, y in zip(a, b):
        dot_product += x * y
        norm_a_sq += x * x
        norm_b_sq += y * y

    norm_a = norm_a_sq ** 0.5
    norm_b = norm_b_sq ** 0.5

    return dot_product / (norm_a * norm_b)

emojis_hdv = {
        '💦': hdv(),
        '🌅': hdv(),
        '🥚': hdv(),
        '🐺': hdv(),
        '🚒': hdv(),
        '🌱': hdv(),
        '🍌': hdv()
        }

steps = [round(i * 0.05, 2) for i in range (21)]

def generate_range_of_hdv_colors(steps):
    k = len(steps) - 1

    vectors = [hdv() for _ in range(k + 1)] # создаем массив из 21 векторов
    for i in range(1, k + 1):

        # пройдемся по элеметам массива HDV
        for j in range(N):

            # 1 / k = 1 / 20 = 0.05, т.е для каждого из элементов шанс того что он поменяет свой знак на противоположный 5 процентов
            # Почему 5 процентов? потому чято расстояния между числами в steps = 0.05
            if random.random() < 1 / k:
                vectors[i][j] = -vectors[i-1][j]
            else:
                vectors[i][j] = vectors[i-1][j]

    return vectors

v = generate_range_of_hdv_colors(steps)

# Попробуем сравнить

value1 = cos_similarity(v[1], v[2])
value2 = cos_similarity(v[1], v[3])

print(value1)
print(value2)

print(value1 > value2)


reds_hdv = generate_range_of_hdv_colors(steps)
greens_hdv = generate_range_of_hdv_colors(steps)
blues_hdv = generate_range_of_hdv_colors(steps)


# Превращается значения steps в индекс
def steps_to_index(v):
    return round(v * 20)

def encode_col(red, green, blue):
    red_hdv = reds_hdv[steps_to_index(red)]
    green_hdv = greens_hdv[steps_to_index(green)]
    blue_hdv = blues_hdv[steps_to_index(blue)]

    return bind(bind(red_hdv, green_hdv), blue_hdv)

def randcol():
    return {
        "r": random.random(),
        "g": random.random(),
        "b": random.random()
    }

ref_colors = [
        {"color": c, "hdv": encode_col(c["r"], c["g"], c["b"])}
        for c in [randcol() for _ in range(1000)]
    ]

def decode_colors(hdv):
    best_color = None
    best_score = -float("inf")

    for item in ref_colors:
        score = cos_similarity(hdv, item["hdv"])

        if score > best_score:
            best_score = score
            best_color = item["color"]

    return best_color

r_color = randcol()
c = encode_col(r_color["r"], r_color["g"], r_color["b"])

print(decode_colors(c))
print(decode_colors(hdv()))

def encode_emoji_col_pair(pair):
    s, c = pair
    return bind(emojis_hdv[s], encode_col(c['r'], c['g'], c['b']))

toy_data1 = [
    ("🍌", randcol()),
    ("💦", randcol()),
    ("🥚", randcol()),
    ("🌅", randcol()),
    ("🌅", randcol()),
    ("🌅", randcol()),
    ("🥚", randcol()),
    ("🐺", randcol()),
    ("🐺", randcol()),
    ("🐺", randcol()),
    ("🌱", randcol()),
    ("🌱", randcol()),
    ("🚒", randcol()),
    ("🥚", randcol()),
    ("💦", randcol()),
    ("🌱", randcol()),
    ("🚒", randcol()),
    ("💦", randcol()),
    ("💦", randcol()),
    ("🍌", randcol()),
]

col_emoji_hdvs = [encode_emoji_col_pair(x) for x in toy_data1]
toy_data_emb = bundle(col_emoji_hdvs)

r1 = decode_colors(bind(toy_data_emb, emojis_hdv['🚒']))
r2 = decode_colors(bind(toy_data_emb, emojis_hdv['💦']))
r3 = decode_colors(bind(toy_data_emb, emojis_hdv['🌱']))
r4 = decode_colors(bind(toy_data_emb, emojis_hdv['🍌']))

print(r1)
print(r2)
print(r3)
print(r4)
