DROP TABLE movie_data
CREATE TABLE movie_data
  (
  title string,
  main_cast string,
  director string,
  genres string,
  lang string,
  plot string,
  imdbURL string
  )
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES(
  "separaterChar" = ",",
  "quoteChar" = '"',
  "escapeChar" = '\\'
STORED AS TEXTFILE
LOCATION '/home/w205user/twitter/imdb/movie_data.csv';
LOAD DATA LOCAL INPATH '/home/w205user/twitter/imdb/movie_data.csv' OVERWRItE INTO TABLE movie_data;
