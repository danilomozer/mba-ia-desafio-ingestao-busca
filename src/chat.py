from search import search_prompt

EXIT_COMMANDS = {"sair", "exit", "quit"}


def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Faça sua pergunta (digite 'sair' para encerrar):\n")

    while True:
        try:
            question = input("PERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAté mais!")
            break

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            print("Até mais!")
            break

        try:
            answer = chain.invoke(question)
        except Exception as exc:  # erro de rede, API key inválida, banco fora do ar etc.
            print(f"Erro ao processar a pergunta: {exc}\n")
            continue

        print(f"RESPOSTA: {answer.strip()}\n")
        print("---\n")


if __name__ == "__main__":
    main()
