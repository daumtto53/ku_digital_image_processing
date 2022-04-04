# 요구사항 :

ㅁGrayscale Transformation 

[ ] 감마값을 최소한 4개 정도로 setting.

[ ] Color image에 대해서도 Option.    

    HSI 변환하고 색조를 유지하는 방법

    RGB 채널에 독립적인 변환을 한 것.

# Negative Transformation

GrayScale.

## transformation 방식

Grayscale : 가져온 (x,y)값을 invert 시킨다.

RGB Color Space : 

1. RGB 채널에 대해서 Invert.

# Power-law transformation

Grayscale.

공식 : S = c * r^gamma

# Histogram Transformation









> http://www.cs.umsl.edu/~sanjiv/classes/cs5420/lectures/color.pdf

> http://www.math.tau.ac.il/~turkel/notes/hsi-to-rgb-conversion.pdf
> 
> 









김은이 교수님 안녕하세요, 디지털영상처리 수강생 201512285 천민수입니다.

Homework #2: point-processing practice 과제 조건에 대해서 확인할 것이 있어 연락드렸습니다.

1. 과제 조건

GrayScale Transformation 

 * 공통

        1.  Grayscale 에 대해 negative 변환

        2.  Color Space 에 대해 negative 변환 (Color image에 대해서도 negative, power law에 대해 RGB 채널 각각에 두 방식(색조유지 / 색조유지 하지 않는 방식)으로 진행할 것.)              

 * Power law Transformation

         감마값을 최소한 4개정도로 setting할 것 

2. 과제 제출 양식

      1. 이전처럼 보고서로 작성하여 제출하면 되는것인지,  
      2. 아니면 보고서 + 변환한 image를 압축하여 과제란에 제출하면 되는것인지  
      3. 보고서 없이 변환한 image를 압축하여 제출하면 되는것인지  

위 큰 틀에서 2가지가 궁금하여 질문드립니다.  

감사합니다. 좋은 하루 되세요.





조교님 안녕하세요, 

Color space의 negative 변환과제를 하던 중 RGB -> HSI -> RGB space로의 변환과정에서 질문이 있어 연락드립니다.



HSI channel을 각각 뽑은 channel을 H, S, I 라고 할때,
Color space의 색조 유지를 하는 negative transformation을 하기 위해 RGB channel 에서   I 만 negative 시킨 I' 를 재료로
H, S, I' channel을 HSI -> RGB 영역으로 아래의 변환공식을 통해 진행한다고 알고있습니다. 



![](C:\Users\MINSOO\AppData\Roaming\marktext\images\2022-03-30-18-46-43-image.png)

<수업 PPT입니다. >



여기서 H는 degree단위고, s는 normalized된 saturity인 것으로 알고 있습니다.

이 변환공식을 코드로 작성하였을 때, 결과 이미지가 다르게 나옵니다.



구현은 아래와 같이 하였고, 





```python
def HSI_to_bgr(h, s, i):
    h = degrees(h)
    if 0 < h <= 120 :
        b = i * (1 - s)
        r = i * (1 + (s * cos(h) / cos(60 - h)))
        g = i * 3 - (r + b)
    elif 120 < h <= 240:
        # h -= 120
        r = i * (1 - s)
        g = i * (1 + (s * cos(h) / cos(60 - h)))
        b = 3 * i - (r + g)
    elif 240 < h <= 360:
        # h -= 240
        g = i * (1 - s)
        b = i * (1 + (s * cos(h) / cos(60 - h)))
        r = i * 3 - (g + b)
```




h는 degree로 계산하였으며

s는 아래와 같이 계산하였습니다.

```python

            b = src[i][j][0] / 255.
            g = src[i][j][1] / 255.
            r = src[i][j][2] / 255. 일 

def rgb_to_saturity(b, g, r):
    if r + g + b != 0:
        return 1 - 3 * np.min([r, g, b]) / (r + g + b)
    else:
        return 0
```



그 결과로 얻어진 r,g,b 값에 정규화한 255를 곱하여



```python

            new_image[i][j][0] = bgr_tuple[0] * 255.
            new_image[i][j][1] = bgr_tuple[1] * 255.
            new_image[i][j][2] = bgr_tuple[2] * 255.
```

new image에다 hsi 를 rgb로 출력한 결과를 출력해보았으나, 



변환 결과가 기대하던 대로 나오지 않아, 혹시 HSI -> RGB 변환에의 공식에 문제가 있지 않았나 싶어 질문드립니다.
