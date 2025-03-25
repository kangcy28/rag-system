FROM mcr.microsoft.com/mssql/server:2022-latest

# Create a config directory
WORKDIR /usr/config

# Copy initialization script
COPY init.sql /usr/config/
COPY setup/ /usr/config/setup/

# Grant permissions for the import script
USER root
RUN chmod +x /usr/config/init.sql

# Switch back to the mssql user
USER mssql

# Command to run when container starts
CMD /bin/bash -c "( sleep 30 ; /opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P $SA_PASSWORD -d master -i /usr/config/init.sql ) & /opt/mssql/bin/sqlservr"