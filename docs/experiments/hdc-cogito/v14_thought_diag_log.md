# v14 thought-loop diagnostic

Does running more thoughts at inference change predictions?

Positions analysed: 114

## Overall (every position)

| n_thoughts | avg top1 | avg true_prob | avg ent | top1 acc | avg rank |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.4033 | 0.0380 | 2.619 | 0.053 | 851.5 |
| 2 | 0.4238 | 0.1882 | 2.732 | 0.228 | 256.5 |
| 3 | 0.3310 | 0.2091 | 3.790 | 0.333 | 205.1 |
| 4 | 0.2018 | 0.1383 | 5.003 | 0.289 | 230.9 |

## Segmented by n=1 uncertainty (true_prob change is the key column)

### very_uncertain  (n=24)

| n_thoughts | avg top1 | avg true_prob |
|---:|---:|---:|
| 1 | 0.1293 | 0.0026 |
| 2 | 0.3307 | 0.2549 |
| 3 | 0.3359 | 0.2524 |
| 4 | 0.2271 | 0.2010 |

### uncertain  (n=52)

| n_thoughts | avg top1 | avg true_prob |
|---:|---:|---:|
| 1 | 0.3288 | 0.0274 |
| 2 | 0.3641 | 0.1263 |
| 3 | 0.2976 | 0.1440 |
| 4 | 0.1739 | 0.0968 |

### confident  (n=38)

| n_thoughts | avg top1 | avg true_prob |
|---:|---:|---:|
| 1 | 0.6783 | 0.0749 |
| 2 | 0.5643 | 0.2308 |
| 3 | 0.3736 | 0.2709 |
| 4 | 0.2240 | 0.1555 |
