# Тестовое задание для MIL-team.

## Описание суперсети

В задании исследуется суперсеть, покрывающая 9 архитектур. В суперсети 2 изменяемых блока и 3 варианта реализации каждого блока. На картинке снизу приведена нумерация реализаций блоков для сокращения.

![Нумерация блоков для различных архитектур](/images/blocks.png)

В ходе выполнения обучалась следующая суперсеть

![Схема суперсети](/images/supernet.png)

В описании сети не уточнено количество каналов в слоях. Я брал 32 между layer1 и первым изменяемым блоком и 64 между layer2 и вторым блоком.

## Обучение сети целиком

Суперсеть была обучена на датасете MNIST. В ходе обучения на каждой итерации семплировалась определенная архитектура и для нее обновлялись веса.

Результаты top-1 acc на тестовой выборке после 10 эпох обучения в процентах (%) приведены в таблице ниже.

| номера блоков |     0 |     1 |     2 |
|---------------|:-----:|:-----:|:-----:|
|             0 | 75.34 | 79.45 | 79.01 |
|             1 | 79.20 | 78.20 | 80.58 |
|             2 | 79.04 | 80.69 | 80.58 |

Слева в строках указана конфигурация первого блока. Сверху в столбцах конфигурация второго блока.

Можно отметить, что нету прямой зависимости между сложностью архитектуры и ее точностью на валидации. Самая сложная архитектура (2, 2) работает хуже архитектуры (2, 1).

![Графики точности на тестовой выборке для разных архитектур](/images/graph_together.jpg)

Как видно, графики обучения похожи по форме, но можно выделить архитектуры (0, 2), (2, 2), которые обучаются быстрее других.

### Ансамблирование

Суперсеть предоставляет возможность доступа ко всем сетям одновременно, что наводит на мысль об их одновременном использовании, то есть составления ансамбля моделей из всех возможных экземпляров суперсети. Метод ансамблирования был добавлен в класс суперсети, ансамбль из вышеописанных моделей доет точность 82.96%, что превышает каждую взятую по отдельности модель.

## Обучение моделей сети по отдельности

В таблице ниже приведены результаты top-1 acc на тестовой выборке после 10 эпох обучения в процентах (%) для каждой из подсетей, обученных по отдельности.

| номера блоков |     0 |     1 |     2 |
|---------------|:-----:|:-----:|:-----:|
|             0 | 96.27 | 96.62 | 97.09 |
|             1 | 96.72 | 96.67 | 96.81 |
|             2 | 96.47 | 96.95 | 97.08 |

![Графики точности на тестовой выборке для разных архитектур](/images/graph_apart.jpg)

Графики обучения похожи по форме. С некоторого момента точность почти не увеличивается, что говорит о пределе обобщающей способности

## Выводы и дальнейшие исследования

Как видно из результатов, обучая архитекруты по отдельности, можно добиться большей точности за то же время. Однако суперсети открывают некоторые перспективы:
- позволяют составлять ансамбли из нейронных сетей, которые, как известно, имеют больший обобщающий потенциал
- суперсети позволяют эффективно сравнивать архитектуры в смысле производительности, что помогает искать оптимальную архитектуру (то есть решает проблему NAS)
- разделение весов позволяет обучать экземпляры, сохраняя оригинальные характеристики, то есть повышается обобщающий потенциал сетей.

