# Requirements

```
pip install ultralytics easyocr pandas opencv-python
```

# Step 1: Generate Dataset
Execute o seguinte comando para gerar um conjunto de imagens. O script irá posicionar aleatoriamente as cartas em diferentes ambientes.

>[!NOTE]
>
> Caso já tenha um dataset pronto, apenas adicione as imagens em uma pasta `/data/images` e as labels em `/data/labels`.
> Crie também o arquivo `classes.txt` com todas as classes do seu dataset (uma em cada linha), e adicione na pasta `/data`,

```
python ./GenerateImagesTrainset.py
```

# Step 2: Split Dataset
Agora, separe o dataset gerado em dois conjuntos: um para treinamento e outro para testes.

## Parâmetros
- `train_pct=0.9`: Define a porcentagem do conjunto de treino (por padrão é 80%).

```
python ./SplitDataset.py --train_pct=0.9
```

# Step 3: Generate `data.yaml` file
Esta etapa cria o arquivo `data.yaml`, que contém informações sobre o caminho dos datasets e as classes.

```
python ./GenerateYamlFile.py
```

# Step 4: Train Model
Agora, treine o modelo com o comando abaixo. O modelo será treinado pelas gerações definidas e o modelo com o melhor desempenho será salvo em /runs/detect/train/weights/best.pt.

## Parâmetros
- `data`: Caminho para o arquivo `.yaml`.
- `yolo11s.pt` Arquivo da versão do YOLO a ser usada.
- `epochs` Define o número de épocas (gerações) para o treinamento.
- `igmsz` Define a resolução das imagens no treinamento. Aumentar este parâmetro melhora a qualidade das imagens, mas também consome mais memória e processamento.


>[!IMPORTANT]
>
>Se você receber a mensagem `Killed` durante o treinamento, é provavelmente devido ao uso excessivo de memória. Uma opção é reduzir o valor de `imgsz` ou realizar o treinamento no [Google Colab](https://colab.research.google.com/drive/1a8Vhs6NA57F_U6_TQLluR_wXyXkC3bkU?usp=sharing), após isso só trazer o modelo zipado e extrair na pasta raíz.
>Se optar por usar o Google Colab, após o treinamento, basta compactar o modelo e transferi-lo para a pasta raiz do seu projeto.

>[!TIP]
>
>Lembre-se de excluir as pastas `images` e `labels` antes de compactar o modelo para enviar para o Google Colab, não vamos mais precisar delas.

```
yolo detect train data=data.yaml model=yolo11s.pt epochs=60 imgsz=640
```