# Requirements

```
pip install ultralytics
```

# Step 1
Execute o comando para gerar um conjunto de dataset, pegando as imagens base de ambientes e das cartas e gerando outras imagens, com a carta posicionada aleatoriamente nesse ambiente.

>[!NOTE]
>
> Caso já tenha um dataset pronto, apenas adicione as imagens em uma pasta `/data/images` e as labels em `/data/labels`.
> Crie também o arquivo `classes.txt` com todas as classes do seu dataset (uma em cada linha), e adicione na pasta `/data`,

```
python ./GenerateImagesTrainset.py
```

# Step 2
Execute o comando para separar o dataset anterior em um conjunto de treinamento e outro conjunto para testes.

## Parâmetros
- `train_pct=0.9` Define o conjunto de treino para 90%, altere conforme necessário.
>[!NOTE]
>
> O padrão do `train_pct` é 80%

```
python ./SplitDataset.py --train_pct=0.9
```

# Step 3
Execute o comando para gerar o arquivo `data.yaml`, contendo informações sobre o path dos datasets e as classes.

```
python ./GenerateYamlFile.py
```

# Step 4
Execute o comando abaixo para treinar o modelo. O modelo é treino pelas quantidade de gerações definidas no comando, e o modelo com a melhor geração é salvo na pasta `/runs/detect/train/weights/best.pt`

## Parâmetros
- `data` Define o path do arquvo `.yaml`;
- `yolo11s.pt` Define a versão do yolo a ser usada;
- `epochs` Define a quantidade de gerações a serem criadas;
- `igmsz` Define a resolução das imagens no treinamento, aumente conforme precisar de mais qualidade na imagem (aumentar esse parâmetro irá consumir mais memória e processamento durante o treinamento);


>[!IMPORTANT]
>
>Caso esteja recebendo `Killed` antes do treinamento ser finalizado, é devido ao uso de memória. Você pode optar por diminuir o parâmetro `igmsz` treinar o modelo pelo [Google Colab](https://colab.research.google.com/drive/1a8Vhs6NA57F_U6_TQLluR_wXyXkC3bkU?usp=sharing), após isso só trazer o modelo zipado e extrair na pasta raíz.

>[!TIP]
>
>Apague as pastas `images` e `labels` antes de compactar, não vamos mais precisar delas, caso precise só rodar o script do passo 1 novamente.

```
yolo detect train data=data.yaml model=yolo11s.pt epochs=60 imgsz=640
```