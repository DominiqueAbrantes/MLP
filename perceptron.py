import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

class Perceptron:

    def __init__(self, X, W, b, epoc, n, yd):
       
        self.X = X
        self.W = W
        self.b = b
        self.epoc = epoc
        self.n = n
        self.yd = yd 

    def ativacao_prever(self, i):
        z = self.X[i][0] * self.W[0] + self.X[i][1] * self.W[1] + self.b
        if z >= 0: 
            yp = 1
        else:
            yp = 0         
        return yp

    def treinamento(self):
        y_yp = [0] * len(self.X)
        erros = [0] * len(self.X)    
        for a in range(1, self.epoc +1 ):
            for i in range(len(self.X)):
                y_yp[i] = self.ativacao_prever(i)
                erros[i] = self.yd[i] - y_yp[i] 
            if not all(erro == 0 for erro in erros):
                for i in range(len(erros)):
                    self.W[0] += self.n * erros[i] * self.X[i][0]
                    self.W[1] += self.n * erros[i] * self.X[i][1]
                    self.b += self.n * erros[i]
                    erros[i] = self.yd[i] - y_yp[i]
            else:
                break             
        return erros

    def avaliar(self, erros):
        erros1 = 0
        for erro in erros:
            if erro != 0:
                erros1 += 1
        acur = (len(self.X) - erros1) / len(self.X) * 100
        return acur 

class Interface_Perceptron(Perceptron):
    def grafico(self, titulo, area_graficos):
        erros = self.treinamento()
        acur = self.avaliar(erros)

        frame = ctk.CTkFrame(area_graficos)
        frame.pack(fill='x', expand=True, padx=10, pady=10)

        resultado = ctk.CTkLabel(
            frame,
            text=(
                f"{titulo}\n\n"
                f"W1: {self.W[0]:.2f}\n"
                f"W2: {self.W[1]:.2f}\n"
                f"b: {self.b:.2f}\n"
                f"Acurácia: {acur:.2f}%"
            ),
            justify='left'
        )
        resultado.pack(side='left', padx=20, pady=20)

        fig = Figure(figsize=(4, 4), dpi=100)
        ax = fig.add_subplot(111)

        for i in range(len(self.X)):
            if self.yd[i] == 0:
                ax.scatter(
                    self.X[i][0],
                    self.X[i][1],
                    color='blue',
                    label='Saída 0' if i == self.yd.index(0) else ''
                )
            else:
                ax.scatter(
                    self.X[i][0],
                    self.X[i][1],
                    color='red',
                    label='Saída 1' if i == self.yd.index(1) else ''
                )

        x = [-0.2, 1.2]

        if abs(self.W[1]) > 0.000001:
            y = [-(self.W[0] * valor + self.b) / self.W[1] for valor in x]
            ax.plot(x, y, color='black', label='Fronteira de decisão')
        elif abs(self.W[0]) > 0.000001:
            x_vertical = [-self.b / self.W[0], -self.b / self.W[0]]
            y_vertical = [-0.2, 1.2]
            ax.plot(x_vertical, y_vertical, color='black', label='Fronteira de decisão')

        ax.set_title(titulo)
        ax.set_xlabel('Entrada X1')
        ax.set_ylabel('Entrada X2')
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.grid(True)
        ax.legend()

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(
            side='right', fill='both', expand=True, padx=10, pady=10
        )


class Rede_XorXnor:

    def __init__(self, titulo, X, yd):
        self.titulo = titulo
        self.X = X
        self.yd = yd

        self.rede = MLPClassifier(
            hidden_layer_sizes=(4,),
            activation='tanh',
            solver='lbfgs',
            max_iter=10000,
            random_state=42
        )

        self.rede.fit(self.X, self.yd)
        self.previsoes = self.rede.predict(self.X)
        self.acur = self.rede.score(self.X, self.yd) * 100


class Interface_rede_XorXnor(Rede_XorXnor):

    def grafico(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill='x', expand=True, padx=10, pady=10)

        resultado = ctk.CTkLabel(
            frame,
            text=(
                f"{self.titulo}\n\n"
                "Camada oculta: 4 neurônios\n"
                f"Acurácia: {self.acur:.2f}%\n"
                f"Previsões: {self.previsoes.tolist()}"
            ),
            justify='left'
        )
        resultado.pack(side='left', padx=20, pady=20)

        fig = Figure(figsize=(4, 4), dpi=100)
        ax = fig.add_subplot(111)

        x = np.linspace(-0.2, 1.2, 200)
        y = np.linspace(-0.2, 1.2, 200)
        xx, yy = np.meshgrid(x, y)
        pontos = np.c_[xx.ravel(), yy.ravel()]

        Z = self.rede.predict_proba(pontos)[:, 1]
        Z = Z.reshape(xx.shape)

        ax.contourf(
            xx,
            yy,
            Z,
            levels=[0, 0.5, 1],
            colors=['blue', 'red'],
            alpha=0.2
        )
        ax.contour(xx, yy, Z, levels=[0.5], colors='black')

        X_grafico = np.array(self.X)
        yd_grafico = np.array(self.yd)

        for classe, cor, nome in [
            (0, 'blue', 'Saída 0'),
            (1, 'red', 'Saída 1')
        ]:
            pontos_classe = X_grafico[yd_grafico == classe]
            ax.scatter(
                pontos_classe[:, 0],
                pontos_classe[:, 1],
                color=cor,
                label=nome
            )

        ax.plot([], [], color='black', label='Fronteira de decisão')
        ax.set_title(self.titulo)
        ax.set_xlabel('Entrada X1')
        ax.set_ylabel('Entrada X2')
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.grid(True)
        ax.legend()

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(
            side='right',
            fill='both',
            expand=True,
            padx=10,
            pady=10
        )

class Rede_Iris(Rede_XorXnor):

    def __init__(self, titulo):

        self.titulo = titulo

        # Carregando o banco de dados Iris
        dados = load_iris()

        self.nome_flores = dados.target_names

        X = dados.data
        yd = dados.target

        # Dividindo os dados em treinamento e teste
        self.X_treinamento, self.X_teste, self.yd_treinamento, self.yd_teste = train_test_split(
            X,
            yd,
            test_size=0.2,
            random_state=42,
            stratify=yd
        )

        # Utilizando a classe Rede_XorXnor
        # para criar e treinar a rede neural
        super().__init__(
            titulo,
            self.X_treinamento,
            self.yd_treinamento
        )

        # Guardando as previsões dos dados de treinamento
        self.previsoes_treinamento = self.previsoes.copy()

        # Realizando as previsões dos dados de teste
        self.previsoes_teste = self.rede.predict(self.X_teste)

        # Calculando a acurácia utilizando apenas os dados de teste
        self.acur = accuracy_score(
            self.yd_teste,
            self.previsoes_teste
        ) * 100


class Interface_rede_Iris(Rede_Iris):

    def interface(self, parent):

        frame = ctk.CTkFrame(parent)
        frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Exibindo a acurácia
        resultado = ctk.CTkLabel(
            frame,
            text=(
                "Classificação das flores Iris\n\n"
                f"Acurácia no teste: {self.acur:.2f}%"
            ),
            font=("Arial", 18),
            justify='left'
        )

        resultado.pack(pady=15)

        # Criando a tabela 
        area_tabela = ctk.CTkScrollableFrame(frame)
        area_tabela.pack(fill='both', expand=True, padx=15, pady=10)

        colunas = (
            "Etapa",
            "Flor real",
            "Previsão da rede",
            "Resultado"
        )

        for coluna, texto in enumerate(colunas):
            ctk.CTkLabel(
                area_tabela,
                text=texto,
                font=("Arial", 14, "bold")
            ).grid(row=0, column=coluna, padx=12, pady=8, sticky='ew')

        linha = 1

        # Inserindo os resultados do treinamento
        for i in range(len(self.yd_treinamento)):

            real = self.nome_flores[self.yd_treinamento[i]]
            previsto = self.nome_flores[self.previsoes_treinamento[i]]

            if self.yd_treinamento[i] == self.previsoes_treinamento[i]:
                situacao = "Correto"
                cor = "green"
            else:
                situacao = "Incorreto"
                cor = "red"

            valores = (
                "Treinamento",
                real,
                previsto,
                situacao
            )

            for coluna, valor in enumerate(valores):
                ctk.CTkLabel(
                    area_tabela,
                    text=valor,
                    text_color=cor if coluna == 3 else None
                ).grid(row=linha, column=coluna, padx=12, pady=4, sticky='ew')

            linha += 1

        # Inserindo os resultados do teste
        for i in range(len(self.yd_teste)):

            real = self.nome_flores[self.yd_teste[i]]
            previsto = self.nome_flores[self.previsoes_teste[i]]

            if self.yd_teste[i] == self.previsoes_teste[i]:
                situacao = "Correto"
                cor = "green"
            else:
                situacao = "Incorreto"
                cor = "red"

            valores = (
                "Teste",
                real,
                previsto,
                situacao
            )

            for coluna, valor in enumerate(valores):
                ctk.CTkLabel(
                    area_tabela,
                    text=valor,
                    text_color=cor if coluna == 3 else None
                ).grid(row=linha, column=coluna, padx=12, pady=4, sticky='ew')

            linha += 1

# dados
y_and = [0, 0, 0, 1]
y_or = [0, 1, 1, 1]
y_xor = [0, 1, 1, 0]
y_nand = [1, 1, 1, 0]
y_nor = [1, 0, 0, 0]
y_xnor = [1, 0, 0, 1]

W1 = [1, -7]
W2 = [2, -0.5]
W3 = [1, 1]
W4 = [-0.5, -0.5]
W5 = [-0.75, -3]

X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

# instanciando objetos
perceptron1 = Interface_Perceptron(X=X, W=W1, b=-0.5, epoc=1000, n=0.1, yd=y_and)
perceptron2 = Interface_Perceptron(X=X, W=W2, b=-2, epoc=1000, n=0.1, yd=y_or)
perceptron3 = Interface_Perceptron(X=X, W=W3, b=-0.5, epoc=1000, n=0.1, yd=y_xor)
perceptron4 = Interface_Perceptron(X=X, W=W4, b=0.7, epoc=100000, n=0.1, yd=y_nand)
perceptron5 = Interface_Perceptron(X=X, W=W5, b=-1, epoc=100000, n=0.1, yd=y_nor)

# janela principal dos perceptrons
janela = ctk.CTk()
janela.title("Resultados do Perceptron")
janela.geometry("1000x750")

area_graficos = ctk.CTkScrollableFrame(janela)
area_graficos.pack(fill='both', expand=True, padx=10, pady=10)

perceptron1.grafico("Problema AND", area_graficos)
perceptron2.grafico("Problema OR", area_graficos)
perceptron3.grafico("Problema XOR", area_graficos)
perceptron4.grafico("Problema NAND", area_graficos)
perceptron5.grafico("Problema NOR", area_graficos)

# janela da rede neural
janela_rede = ctk.CTkToplevel(janela)
janela_rede.title("Resultados da Rede Neural")
janela_rede.geometry("1000x750+1020+20")

area_graficos_rede = ctk.CTkScrollableFrame(janela_rede)
area_graficos_rede.pack(fill='both', expand=True, padx=10, pady=10)

rede_xor = Interface_rede_XorXnor("Problema XOR", X, y_xor)
rede_xor.grafico(area_graficos_rede)

rede_xnor = Interface_rede_XorXnor("Problema XNOR", X, y_xnor)
rede_xnor.grafico(area_graficos_rede)

# janela da classificação Iris
janela_iris = ctk.CTkToplevel(janela)
janela_iris.title("Classificação das Flores Iris")
janela_iris.geometry("1000x750+20+20")
janela_iris.update()
janela_iris.deiconify()
janela_iris.lift()
janela_iris.focus()

iris = Interface_rede_Iris("Rede Neural - Iris")
iris.interface(janela_iris)

janela.mainloop()