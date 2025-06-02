from langchain_community.tools import ShellTool

cmd_tool = ShellTool()
# using this we can execute any commands that we need to execute in our cmd or powershell
results = cmd_tool.invoke('dir')
print(f'List of all the files in directory:  {results}')
# cmd_tool.invoke('mkdir dummy')
cmd_tool.invoke('rmdir dummy')