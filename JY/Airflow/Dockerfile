# ----------------------------------------------------------------------
# ✅ Base image: Apache Airflow official image
# ----------------------------------------------------------------------
FROM apache/airflow:2.6.1

# ----------------------------------------------------------------------
# 🧩 Switch to root to install OS-level dependencies
# ----------------------------------------------------------------------
USER root
ENV DEBIAN_FRONTEND=noninteractive

# Install Java (OpenJDK 17), procps (for ps), and bash
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-17-jdk-headless \
        procps \
        bash \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    # Ensure bash is the default shell (Airflow’s entrypoint scripts use /bin/sh by default)
    ln -sf /bin/bash /bin/sh && \
    # Create expected JAVA_HOME dir and symlink the java binary
    mkdir -p /usr/lib/jvm/java-17-openjdk-amd64/bin && \
    [ -f /usr/lib/jvm/java-17-openjdk-amd64/bin/java ] || ln -s "$(which java)" /usr/lib/jvm/java-17-openjdk-amd64/bin/java

# Environment setup for Spark or Java-based tools
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# ----------------------------------------------------------------------
# 📦 Install Python dependencies
# ----------------------------------------------------------------------
# Copy requirements.txt into the container
COPY requirements.txt /requirements.txt

# Switch to airflow user for pip install (good practice)
USER airflow

RUN pip install --no-cache-dir -r /requirements.txt

# ----------------------------------------------------------------------
# 🗂️ Directory setup for your project layout
# ----------------------------------------------------------------------
# Create paths that will be mounted via docker-compose
RUN mkdir -p /opt/airflow/{dags,utils,config,data,model_store,datamart,logs}

# Working directory (optional but harmless)
WORKDIR /opt/airflow

# ----------------------------------------------------------------------
# ✅ Default command handled by docker-compose (webserver/scheduler)
# ----------------------------------------------------------------------
