from Trainer import train_regraft

if __name__ == "__main__":
    input_path = '../data'
    output_path = '../results'
    train_regraft(TEST=True, input_path=input_path, output_path=output_path)