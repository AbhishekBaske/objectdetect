import labelbox
from labelbox.schema.project import Project
labelbox_client = labelbox.Client(api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbGVsdzF1Y3EybzdmMDd6ZzhhNmJhZms1Iiwib3JnYW5pemF0aW9uSWQiOiJjbGVsdzF1YzkybzdlMDd6Z2FiaGkybXZ1IiwiYXBpS2V5SWQiOiJjbGVseXU1bjI1bnduMDd6YWd6Nm03bnE0Iiwic2VjcmV0IjoiMDg5OGJiYWMyNTg5M2E1MmQ4MWIyOGZlMmE2NDdjZWQiLCJpYXQiOjE2Nzc0NTA3MDIsImV4cCI6MjMwODYwMjcwMn0.i6ej0e42YcsS7SupDwtg1sSxAMhDd1EKKFurl-lBqHk")
project = labelbox_client.get_project('clelwkf7531fs07xp5l9572oj')

print(project)