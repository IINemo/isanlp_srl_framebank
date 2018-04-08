#!/bin/bash

echo 
SetEnv PYTHONPATH /src

printf "\nSetEnv IP_ADDRESS $IP_ADDRESS" >> /etc/apache2/sites-enabled/demo.conf
printf "\nSetEnv MORPH_PORT $MORPH_PORT" >> /etc/apache2/sites-enabled/demo.conf
printf "\nSetEnv SYNTAX_PORT $SYNTAX_PORT" >> /etc/apache2/sites-enabled/demo.conf
printf "\nSetEnv SEM_PORT $SEM_PORT" >> /etc/apache2/sites-enabled/demo.conf

$@
