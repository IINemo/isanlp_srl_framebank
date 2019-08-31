FROM inemo/isanlp_base

RUN apt-get clean && apt-get update && apt-get install -y apache2 python
RUN a2enmod cgi

COPY isanlp_srl_framebank /src/isanlp_srl_framebank

COPY demo /var/www/demo
RUN chmod +x /var/www/demo/wui.py

COPY brat /var/www/demo/brat

EXPOSE 80

COPY apache_configs/* /etc/apache2/sites-enabled/
RUN echo export PATH=/root/.pyenv/shims:/root/.pyenv/bin:$PATH >> /etc/apache2/envvars

COPY entrypoint.sh /entrypoint.sh

RUN chmod 777 -R /root/.pyenv && chmod 777 /root

RUN pip install -e /src/isanlp_srl_framebank

ENTRYPOINT ["/entrypoint.sh"]
CMD ["apache2ctl", "-D", "FOREGROUND"]
