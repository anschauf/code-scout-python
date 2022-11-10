
# while IFS== read -r key value; do
#   printf -v "$key" %s "$value" && export "$key"
# done <./.env

export $(xargs <./.env)

#while read -r line
#  do
#    echo $line
#done < ./.env