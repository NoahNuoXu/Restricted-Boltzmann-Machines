import sys

import pandas as pd



def main(rating_file, output_file):
    df = pd.read_csv(rating_file)
    print(len(set(df['user_id'])))
    print(df['user_id'].max())
    print(len(set(df['anime_id'])))
    print(df['anime_id'].max())
    df = df[df['rating'] > df['rating'].mean()]
    df.drop(columns=['rating'], inplace=True)
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    rating_file = sys.argv[1]
    output_file = sys.argv[2]
    main(rating_file, output_file)
