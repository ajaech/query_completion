import fileinput

current_user = None
current_query = None
current_time = None

for line in fileinput.input():
  if not fileinput.isfirstline():
    fields = line.split('\t')
    user_id = fields[0]
    query = fields[1]
    query_time = fields[2]

    if current_query != query:
      if current_query is not None:
        print '\t'.join((current_user, current_query, current_time))
      current_query = query
      current_user = user_id
      current_time = query_time

print '\t'.join((current_user, current_query, current_time))
    
