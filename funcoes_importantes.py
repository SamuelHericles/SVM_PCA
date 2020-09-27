"""
                      Universidade Federal do Ceará
                            Campus - Sobral
                     Curso de Engenharia da Computação
                   Disciplina de Reconhecimento de Padrões
                             4º TRABALHO
                     SAMUEL HERICLES SOUZA SILVEIRA

Este arquivo consta as funções importantes ou base para programação dos algoritmos requisitados.
"""

# Imports necessários
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


"""
    Carrega a base de dados
Esta classe carrega a base de dados com que fiz um link raw no github para não me preocupar com a localização deste no repositório, como
também caso vá ser executado no google colab não precisão dos imports do google.path para carregar os arquivos. Com isso, além do load, 
eu organizo os dados pela o atributo da ultima coluna no qual esta são os rótulos das amostras.
"""
class carrega_base:
    caminho = 'https://raw.githubusercontent.com/SamuelHericles/Algoritmos_de_classificao_baysianos/master/dema_dados.csv'
    derma_dados = pd.read_csv(caminho)
    derma_dados.sort_values('c35',inplace=True)
    derma_dados.reset_index(drop=True,inplace=True)

    
"""
    Classe das funções bases ou principais
    
Nesta classe temos as funções necessárias para que possa fazer o código dos algoritmos de classificação mais limpo.
"""    
class funcoes_main:
    
    
    # Função para for carregada a classe funcoes_main já carrega a base dados
    def __init__(self):
        self.base = carrega_base.derma_dados

    # Calcula a média de cada atributo e armazena em um vetor, neste caso, o vetor é um dataframe de uma linha só e com 34 colunas
    def vetor_medio(self,base):
        vetor_medio = []
        for i,j in zip(base.columns[:-1],[i for i in range(base.shape[1])]):
            vetor_medio.append(base[str(i)].mean())
        df = pd.DataFrame(data=vetor_medio,index=base.columns[:-1])
        return df.T


    # Calcula a variância de cada atritubo e armazena em um vetor, neste caso o vetor é um dataframe de uma linha só e com 34 colunas
    def vetor_variancias(self,base):
        vetor_medio = []
        for i,j in zip(base.columns[:-1],[i for i in range(base.shape[1])]):
            vetor_medio.append(base[str(i)].var())
        df = pd.DataFrame(data=vetor_medio,index=base.columns[:-1])
        return df.T

    # Calcula a covariância de um atributo com outro(ou ele mesmo sendo isso a variância) e armazena em um dataframe matriz de 34x34
    def matriz_covariancia(self,base):
        
        #     Cria um dataframe do tamanho do número de atributos na base, sendo o ultimo atributo os rótulos das amostras, logo
        #é retirados do cáluclo, por isso base.columns[:-1]
        df = pd.DataFrame(index=[base.columns[i] for i in range(base.shape[1])],columns=base.columns[:-1])
        
        #     São dois for mas poderiam ser 4, mas existe a função zip que faz com o for percorra um dois ou mais vetores, com isso
        #eu posso percorrer as colunas da base de dados e ao mesmo tempo que faço a covariância dos atributos populo a matriz de
        #covariância.
        for i,count_i in zip(base.columns[:-1],[i for i in range(base.shape[1])]):
            for j,count_j in zip(base.columns[:-1],[i for i in range(base.shape[1])]):
                
                # Calcula a covariância
                df.iloc[count_i,count_j] = (sum(base[str(i)]*base[str(j)])- 
                                            (sum(base[str(i)])*sum(base[str(j)]))/base.shape[0])/base.shape[0]
                
        # Retorna uma coluna a menos pois é a coluna de classes e ela está como NaN
        return df.iloc[:-1]

    
    # Normalização z-score, calculo a media e o desvio padrão de um atributo e normalizo os dados dele.
    # Com o detalhe que a normalização da base de teste são com a media e desvio padrão dos dados de trenio
    def norm_z_score(self,base_trenio, base_teste):
        
        # Média
        media         = base_trenio[base_trenio.columns[:-1]].mean()
        
        # Desvio padrão
        desvio_padrao = base_trenio[base_trenio.columns[:-1]].std()
        
        # Faço a normalização da base de trenio
        base_trenio[base_trenio.columns[:-1]] = base_trenio[base_trenio.columns[:-1]]-media/desvio_padrao
        base_teste[base_teste.columns[:-1]]   = base_teste[base_teste.columns[:-1]]-media/desvio_padrao
        
        # Retorna das duas bases normalizada
        return base_trenio,base_teste

    
    # K-fold para treinamento, aqui foi feito com que pegasse 20% dos dados aleatoriamente e criassem a base de teste e 
    # o que sobrou a base de treino. Não estratifiquei pois há muitas amostras em cada classe então há um chance muita alta
    # tenha amostras de todas as classes em cada base.
    def kfold_shuffle(self,k=5):
        
        # Cria uma dataframe da treino e teste
        X = pd.DataFrame({})
        y = pd.DataFrame({})

        # Pego 20% dos indices da base de dados aleatoriamente e retiro estes para colocar na base de teste
        # o que sobrou coloco na base trenio
        index_teste_classe = sorted(np.random.choice(self.base.index.values,round((self.base.shape[0]/k))))
        X_classe = self.base.iloc[[i for i in self.base.index if i not in index_teste_classe],:]
        y_classe = self.base.iloc[index_teste_classe,:]

        # Quando feito a armazenada em X e y, os indices vão está os mesmo da antiga base, então eu reseto eles.
        X = X.append(X_classe,ignore_index=True)
        y = y.append(y_classe,ignore_index=True)

        # Com isso, como foi requisitado eu já normalizo os dados na saida
        return self.norm_z_score(X,y)

    
    # Calculo a probabilidade a priori de cada classe de uma base dados.
    def prob_a_priori(self,base):
        probs_p = []
        for i in base['c35'].unique():
            probs_p.append(base.query('c35=='+str(i))['c35'].shape[0]/base.shape[0])
        return probs_p

    
    # Aqui é feito o teste de um elemento da base de teste, esta função serve para o algoritmo QDA e Naive Bayes
    def teste_elem(self,y,X_medio,mcov_inv,determinante,probabilidade):
          
        # Corrigo a dimensão da base, pois algumas em python vem como (34,) ou (5,) então corrigo para (34,1) ou (5,1)
        y = y.values.reshape((1,-1))
        X_medio = X_medio.values.reshape((-1,1))
        
        # Calculo o logaritmo natural do determinate da matriz de covariância inversa
        det_log  = np.log(abs(determinante))
        
        # Crio a matriz linha ou vetor linha dos dados de teste com o vetor médio de cada classe da base de treino
        m_linha  = (y-pd.DataFrame(X_medio).T).values
       
        # Crio a matriz coluna ou vetor coluna dos dados de teste com o vetor médio de cada classe da base de treino
        m_coluna = (y-pd.DataFrame(X_medio).T).T.values
        
        # Calculo o logaritmo da probabilidade a priori
        log_prob = np.log(probabilidade)
        
        # Realizo o teste para projeção dos dados de teste, aqui o retorno eh uma escala que quanto menor for é o rotulo exato no 
        # qual o algoritmo retorna
        resultado = det_log + (m_linha @ mcov_inv) @ (m_coluna) - 2*log_prob
        return resultado[0][0]


    # Aqui é feito o calculo do algoritma do lda, no qual testa dos dados de teste com a projeção dos dados de treino
    def teste_elem_lda(self,y,X_medio,mcov_inv):
        
        y = y.values.reshape((1,-1))
        X_medio = X_medio.values.reshape((-1,1))
        
        # Crio a matriz linha ou vetor linha dos dados de teste com o vetor médio de cada classe da base de treino
        m_linha   = (y-X_medio.T)
        
        # Crio a matriz coluna ou vetor coluna dos dados de teste com o vetor médio de cada classe da base de treino
        m_coluna  = (y-X_medio.T).T
        
        # Calculo a projeção dos dados para retornar um escala no qual o menor deste é referente a classe que o algoritmo retorna
        resultado = (m_linha @ mcov_inv) @ m_coluna
        return resultado[0][0]

    # Calcula a acurácia dos resultados
    def get_acc(self,y_pred,y_true):
        return round(sum(y_pred==y_true)/y_true.shape[0],4)    
    
    # Limpa a covoriância da matriz de covariância para aplicar o algoritmo naive bayes pois ele considera que não covariância entre
    # os dados pois eles são evento indepentes entre eles.
    def limpar_covariancia(self,matriz_covariancia):
        for i in range(matriz_covariancia.shape[0]):
            for j in range(matriz_covariancia.shape[0]):
                
                # Zero os dados que não estão na diagonal, ou seja, só há variância dos atributos, não covariância
                if i!=j:
                    matriz_covariancia.iloc[i,j]=0
        return matriz_covariancia
    
    # Faço a correção lambda, é preciso dela pois adiciono um valor muito pequeno a matriz de covariância para que haja um determinante
    # e com isso uma matriz inversa.
    def correcao_lambda(self,matriz_de_covariancia):
         return matriz_de_covariancia + np.identity(matriz_de_covariancia.shape[0], dtype=float)*0.01
    
    # Calcula a inversa da matriz de covariância
    def matriz_inversa(self,matriz_covariancia):
         return pd.DataFrame(np.linalg.inv(np.matrix(matriz_covariancia.values, dtype='float')))
        
        
    # Calcula o determinante da matriz de covariância ou a inversa dela.
    def determinante_matriz_covariancia(self,matriz_covariancia):
        return np.linalg.det(np.matrix(matriz_covariancia.values, dtype='float'))        
    
    # Calcula o vetor médio de todas as classes
    def vetor_medio_p_classe(self,base):
        vt_medios = pd.DataFrame(index=base.columns[:-1])
        for i in sorted(base[base.columns[-1]].unique()):
            vt_medios['classe'+str(i)] = self.vetor_medio(base.query('c35=='+str(i))).T
        return vt_medios.T
    
    # Calculo a matriz de correlação dos atributos
    def matriz_correlacao(self,base,vt_medio):
        
        # Cria df para popular
        df = pd.DataFrame(index=[base.columns[i] for i in range(base.shape[1]-1)],columns=base.columns[:-1])
        
        # Calcula a correlação de um atributo com outro(pode ser ele mesmo) e popula o df
        for i,count_i in zip(base.columns[:-1],[i for i in range(base.shape[1])]):
            for j,count_j  in zip(base.columns[:-1],[i for i in range(base.shape[1])]):

                # Numerador que é a covarância entre os atributos
                num = np.sum(np.dot((base[i]-vt_medio[i].values),
                                    (base[j]-vt_medio[j].values)))

                # Denominador calculo a variância de um atributo vezes a variância de outro
                dem = np.sqrt(np.dot(np.sum(pow(base[i]-vt_medio[i].values,2)),
                                     np.sum(pow(base[j]-vt_medio[j].values,2))))

                df.iloc[count_i,count_j] = num/dem
        return df
    
    # Faço a projeção do vetor médio menos com a média dos vetores médio
    def calcula_projecao(self,base,u,u_classe):
        return pd.DataFrame(base.shape[0]*((u_classe - u).values*(u_classe - u).T.values))
   
    # Divido os dados entre 70% para treino e 30% para teste para abordagem CDA
    def treino_teste_70_30(self,base):
        X = pd.DataFrame({})
        y = pd.DataFrame({})

        index_teste_classe = sorted(np.random.choice(base.index.values,int(base.shape[0]*0.3)))
        X_classe = base.iloc[[i for i in base.index if i not in index_teste_classe],:]
        y_classe = base.iloc[index_teste_classe,:]

        X = X.append(X_classe,ignore_index=True)
        y = y.append(y_classe,ignore_index=True)

        return X,y
    
    # Transformo os dados a partir da matriz W do CDA
    def transformar_dados(self,W,base):
        Xtr_t = pd.DataFrame(base[base.columns[:-1]].values @ W.T.values).T
        Xtr_t['c35'] = base['c35']
        
        return Xtr_t
    
    # Plota a correlação entre os dados a partir de um gráfico heatmeap
    def plot_matriz_de_correlacao(self,base_derma):
        colunas = {
    'c1':'Eriteme','c2':'Escala','c3':'Bordas definidas','c4':'Coceira','c5':'fenômeno de Koebner','c6':'Pápulas poligonais',
    'c7':'Envolvimento da mucosa oral','c8':'Envolvimento do joelho e do cotovelo','c9':'Envolvimento do escalpo',
    'c10':'Incotinência de melaninca','c11':'Eosinófilos no infiltrado','c12':'Infriltrado PNL','c13':'Fibrose na derme papilar',
    'c14':'Exocitose','c15':'Acantose','c16':'Hiperceratose','c17':'Paracertose','c18':'Dilatação',
    'c19':'Dilatação em clava dos cones epiteliais','c20':'Dilatação em elava dos cones epiteliais',
    'c21':'Alongamento dos cones epiteliais da epiderme','c22':'Pústulas espongiformes',
    'c23':'Microabscesso de Munro','c24':'Hipergranulose focal','c25':'Ausência da camada granulosa',
    'c26':'Vacuolização e destruição da camada basal','c27':'Espongiose','c28':'Aspecto dente de serra das cristas interpapilares',
    'c30':'Tampões cárneos foliculares',
    'c29':'Tampões cárneos foliculares','c31':'Paraceratose perifolicular','c32':'Infiltrado inflamatório mononuclear',
    'c33':'Infiltrado em banda','c34':'Idade','c35':'Classes',
                }
        corr = base_derma.rename(columns=colunas).corr()
        plt.figure(figsize=(20,10))
        ax = sns.heatmap(corr)
    
    # Matriz de confusão, usa as funções abaixo para produzir um plot adequado
    def matriz_confusao(self,y_true,y_pred):
        df_resultado = pd.DataFrame(data=[y_true.values,y_pred]).T
        df_resultado.rename(columns={0:'Esperado',1:'Previsto'},inplace=True)
        df_m_confusao = pd.DataFrame(index=[i for i in y_true.unique()],columns=y_true.unique())
        for i in range(df_m_confusao.shape[0]):
            for j in range(df_m_confusao.shape[0]):
                df_m_confusao.iloc[i,j] = df_resultado.query('Esperado=='+str(i+1)+' and Previsto=='+str(j+1)).sum()[0]
        return df_m_confusao

    # Calcula a precisão dos dados
    def precisao(self,TP,FP):
        return TP/(FP+TP)
    
    # Calcula o recall dos dados
    def recall(self,TP,FN):
        return TP/(TP+FN)
    
    # Calcula a F1-score dos dados
    def F1(self,precisao,recall):
        return (2*precisao*recall)/(precisao+recall)
    
    # Calcula a acurácia dos dados 
    def acuracia(self,TP,TN,FN,FP):
        return (TP+TN)/(TP+TN+FN+FP)

    # Exibe cada parâmetro a após a execução do modelo
    def parametros_de_medicao(self,df_m_confusao):
        metricas = []
        for classe in range(df_m_confusao.shape[0]):
            TP = df_m_confusao.iloc[classe,classe]
            valores = [i for i in range(6) if i!=classe]
            FP = np.sum(df_m_confusao.iloc[classe,valores])
            TN = np.sum([df_m_confusao.iloc[i,i] for i in range(6) if i!=classe])
            FN = np.sum(df_m_confusao.iloc[valores,classe])

            resultados = {
                'precisao':self.precisao(TP,FP),
                'recall'  :self.recall(TP,FN),
                'acurácia':self.acuracia(TP,TN,FN,FP),
                'f1-score':self.F1(self.precisao(TP,FP),self.recall(TP,FN))
            }

            metricas.append(resultados)

        return pd.DataFrame(metricas).mean()
    
    # Plota o resultados das funções acima em um heatmap
    def plot_matiz_de_confusao(self,df_m_confusao):
        classes={1:'Classe 1',2:'Classe 2',3:'Classe 3',4:'Classe 4',5:'Classe 5',6:'Classe 6'}
        df_m_confusao.rename(columns=classes,index=classes,inplace=True)

        group_counts = ["{0:0.0f}".format(value) for value in df_m_confusao.values.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in df_m_confusao.values.flatten()/np.sum(df_m_confusao.values)]

        labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]

        labels = np.asarray(labels).reshape(6,6)
        plt.figure(figsize=(10,6))
        sns.heatmap(df_m_confusao.astype(float), annot=labels, fmt='', cmap='Blues');
        plt.title('Matriz de confusão')
        plt.show()
        
    # Classificação com multi-classe: 1 vs all, quem for da classe 1(por exemplo) fica com 1 e o resto com -1
    def classe_1_menos_1(base,classe):
        y = pd.concat([pd.DataFrame(np.ones([base.query('c35 == '+str(classe)).shape[0],1])),
                       pd.DataFrame(-np.ones([base.query('c35 != '+str(classe)).shape[0],1]))])
        y.reset_index(drop=True,inplace=True) 
        return y

    # Ordenar a base dados pelos rótulos
    def ordena_por_classe(base):
        base.sort_values('c35',inplace=True)
        base.reset_index(drop=True,inplace=True)
        return base

    # A base de dados urban é muito desbalanceada, então fiz a normalização max-min para deixar valores entre {0,1}
    def normaliza_max_min(base):
        aux = base.iloc[:,:-1]
        aux = (aux - aux.min())/(aux.max()-aux.min())
        aux['c35'] = base['c35']
        return aux
    
    # Pegar a acurácia da base urban, a getacc antes era para pegar a acuráica para o trabalho anterior de LDA.
    def acc(y_true, y_pred):
        return round(sum(y_pred==y_true.values[:,0])/y_pred.shape[0],2)        
        