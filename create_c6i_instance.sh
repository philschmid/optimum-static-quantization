NAME=optimum

# create key pair for ssh
aws ec2 create-key-pair --key-name ${NAME} --query 'KeyMaterial' --output text > ${NAME}.pem
chmod 400 ${NAME}.pem

# create security group
DEFAULT_VPC_ID=$(aws ec2 describe-vpcs --query 'Vpcs[?isDefault==true].VpcId' --output text)
echo "Default VPC ID: ${DEFAULT_VPC_ID}"
SECURITY_GROUP_ID=$(aws ec2 create-security-group --group-name ${NAME}-sg --description "SG for Optimum tests" --vpc-id ${DEFAULT_VPC_ID} --output text)
echo "Security Group ID: ${SECURITY_GROUP_ID}"
echo $(aws ec2 authorize-security-group-ingress --group-id ${SECURITY_GROUP_ID} --protocol tcp --port 22 --cidr 0.0.0.0/0 --output text)

# start instance
AMI_ID=$(aws ec2 describe-images --filters "Name=name,Values=*Ubuntu Server 20.04 LTS*"  --query 'Images[0].ImageId' --output text)
echo "AMI ID: ${AMI_ID}"
INSTANCE_TYPE=c6i.xlarge

aws ec2 run-instances \
  --image-id ${AMI_ID} \
  --key-name ${NAME} \
  --count 1 \
  --instance-type ${INSTANCE_TYPE} \
  --security-group-ids ${SECURITY_GROUP_ID} \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${NAME}-demo}]" \
  2>&1 > /dev/null

# connect via ssh
echo "waiting for the instance to start..."
sleep 45

PUBLIC_DOMAIN=$(aws ec2 describe-instances \
    --filters Name=tag-value,Values=${NAME}-demo  \
    --query 'Reservations[*].Instances[*].PublicDnsName' \
    --output text)

echo "connect to instance via ssh with:\nssh -i ${NAME}.pem ubuntu@${PUBLIC_DOMAIN//[$'\t\r\n ']}"


