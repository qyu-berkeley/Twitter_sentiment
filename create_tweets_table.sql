
drop table if exists tweets;

create table tweets (text varchar(250) PRIMARY KEY NOT NULL,
movie varchar(50),
language varchar(50),
country varchar(50),
user_nm varchar(50),
screen_nm varchar(50),
coordinates_lat varchar(50),
coordinates_long varchar(50),
location varchar(50),
retweets_count INTEGER,
followers_count INTEGER,
favourites_count INTEGER,
friends_count INTEGER,
text_clean varchar(250),
sentiment_score float,
sentiment varchar(50));



ALTER TABLE tweets ADD COLUMN created_at TIMESTAMP;
ALTER TABLE tweets ALTER COLUMN created_at SET DEFAULT now();

drop table if exists tweet_words;

create table tweet_words (word varchar(50) PRIMARY KEY NOT NULL, movie varchar(40),
count integer, word_sentiment varchar(50));
