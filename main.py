from utils.processing_datasets import *
from utils.base_functions import *
from utils.get_distribution import *
import sys

def main():
    print("\nFerramenta de Análise de Dados de Chuva\n")
    print("Escolha uma opção:")
    print("1. Já baixei os datasets")
    print("2. Ainda não baixei os datasets")

    try:
        escolhaIniciar = int(input("Digite sua escolha (1 ou 2): "))

        if escolhaIniciar == 2:
            print("\nSiga estas instruções para baixar os datasets:\n")

            print("Dados do CEMADEN:\n")
            print("1. Acesse o mapa interativo do CEMADEN: https://mapainterativo.cemaden.gov.br/#")
            print("2. Selecione a região e o período desejados para download.")
            print("3. Baixe os datasets para cada mês necessário.")
            print("4. Organize os arquivos na seguinte estrutura de pastas:")
            print("   datasets/cemaden/")

            print("\nDados do INMET:\n")
            print("1. Acesse o site do INMET: https://portal.inmet.gov.br/dadoshistoricos")
            print("2. Selecione o ano desejado.")
            print("3. Baixe o arquivo, identifique a estação desejada no .zip e organize na seguinte estrutura de pastas:")
            print("   datasets/inmet/")

            print("\nApós baixar e organizar os datasets, execute este script novamente e escolha a opção 1.")
        elif escolhaIniciar == 1:
            print("\nÓtimo! Certifique-se de que seus datasets estão organizados na seguinte estrutura:")
            print("datasets/cemaden/ (para os datasets do CEMADEN)")
            print("datasets/inmet/ (para os datasets do INMET)")
            
            print("\nVocê deseja analisar quais dataframes?\n")
            print("1. INMET")
            print("2. CEMADEN")
            print("3. Ambos")
            
            try:
                escolhaDataframe = int(input("Digite sua escolha (1, 2 ou 3): "))
                
                if escolhaDataframe == 1:
                    aut, conv = process_data(DataSource.INMET, 'datasets/')
                elif escolhaDataframe == 2:
                    jd, cj, av = process_data(DataSource.CEMADEN, 'datasets/')
                elif escolhaDataframe == 3:
                    aut, conv = process_data(DataSource.INMET, 'datasets/')
                    jd, cj, av = process_data(DataSource.CEMADEN, 'datasets/')                     
                else:
                    print("\nOpção inválida. Por favor, execute o script novamente e digite 1, 2 ou 3.")
                    sys.exit()
                    
                aggregate_to_csv(aut, 'inmet_aut')
                aggregate_to_csv(conv, 'inmet_conv')
                aggregate_to_csv(jd, 'cemaden_jardim')
                aggregate_to_csv(cj, 'cemaden_cidade')
                aggregate_to_csv(av, 'cemaden_agua')
                
                process_precipitation_series(['inmet_aut', 'inmet_conv', 'cemaden_jardim', 'cemaden_cidade', 'cemaden_agua'],'daily')
                
                
            except ValueError:
                print("\nErro: Por favor, digite um número válido (1, 2 ou 3). Execute o script novamente.")
                sys.exit()
        else:
            print("\nOpção inválida. Por favor, execute o script novamente e digite 1 ou 2.")
    except ValueError:
        print("\nErro: Por favor, digite um número válido (1 ou 2). Execute o script novamente.")
        sys.exit()

if __name__ == "__main__":
    main()


