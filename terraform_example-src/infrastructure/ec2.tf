resource "random_pet" "instance_name" {
  count = var.instance_count
  length = 2 // You can adjust the number of words in the random name
}

resource "aws_instance" "app_server" {
  ami           = var.ami_id
  instance_type = var.instance_type
  subnet_id     = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.ec2_sg.id]
  iam_instance_profile = aws_iam_instance_profile.ec2_profile.name
  key_name = "test_terraform" # Use the existing key pair
  user_data = file("user_data.sh") # Add this line
  count = var.instance_count

  tags = {
    Name = "${var.project_name}-instance-${random_pet.instance_name[count.index].id}"
  }
}