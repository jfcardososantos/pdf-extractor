#!/bin/bash

# Verificar se o Docker está instalado
if ! command -v docker &> /dev/null; then
    echo "Docker não encontrado. Instalando Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
fi

# Verificar se o Docker Compose está instalado
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose não encontrado. Instalando Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Verificar se o NVIDIA Container Toolkit está instalado
if ! command -v nvidia-container-toolkit &> /dev/null; then
    echo "NVIDIA Container Toolkit não encontrado. Instalando..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update
    sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker
fi

# Criar diretório para arquivos temporários
mkdir -p temp

# Construir e iniciar o container
echo "Construindo e iniciando o container..."
docker-compose up -d --build

# Configurar para iniciar com o sistema
echo "Configurando para iniciar com o sistema..."
sudo tee /etc/systemd/system/pdf-extractor.service << EOF
[Unit]
Description=PDF Extractor Docker Container
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$(pwd)
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

# Recarregar systemd e habilitar o serviço
sudo systemctl daemon-reload
sudo systemctl enable pdf-extractor.service

echo "Instalação concluída! O serviço iniciará automaticamente com o sistema."
echo "Para verificar o status, use: sudo systemctl status pdf-extractor"
echo "Para ver os logs, use: docker-compose logs -f" 