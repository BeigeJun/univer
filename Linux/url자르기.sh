#!/bin/bash

url="ftp://squid-game.com:456/game?name=무궁화꽃"
#url="https://www.linux.com:8080/search?key=LOVE"

protocol=${url%%:*}

host_1=${url#*//}
host=${host_1%:*}

port_1=${url##*:}
port=${port_1%/*}

app_1=${url##*/}
app=${app_1%'?'*}

query=${url#*'?'}

echo "protocol : $protocol"
echo "host : $host"
echo "port : $port"
echo "app : $app"
echo "query : $query"
