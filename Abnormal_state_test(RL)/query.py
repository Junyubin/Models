def normal_query(start_date, end_date, interval):
    sql = """
    select
        toStartOfInterval(logtime, INTERVAL {interval}) as lgtime, src_ip, dst_ip,

        arrayStringConcat(groupUniqArray(replaceRegexpAll(replace(decodeURLComponent(http_host), '/..', ' pathsearcherdetected '), '[\-%./!@#$?,;:&*)(+=0-9_]', ' ')), ' ') as host,
        arrayStringConcat(groupUniqArray(replaceRegexpAll(replace(decodeURLComponent(http_agent), '/..', ' pathsearcherdetected '), '[\-%./!@#$?,;:&*)(+=0-9_]', ' ')), ' ') as agent,
        arrayStringConcat(groupUniqArray(replaceRegexpAll(replace(decodeURLComponent(http_query), '/..', ' pathsearcherdetected '), '[\-%./!@#$?,;:&*)(+=0-9_]', ' ')), ' ') as query,
        'normal' as label
    
    from dti.dti_sh_demo_log
        where (logtime >= '{start_date}' and logtime < '{end_date}')
            and hash == 'normal'
        group by lgtime, src_ip, dst_ip
    """.format(interval=interval, start_date=start_date, end_date=end_date)
    
    return sql

def attack_query(attack, start_date, end_date, interval):
    sql = """
    select
        toStartOfInterval(logtime, INTERVAL {interval}) as lgtime, src_ip, dst_ip,

        arrayStringConcat(groupUniqArray(replaceRegexpAll(replace(decodeURLComponent(http_host), '/..', ' pathsearcherdetected '), '[\-%./!@#$?,;:&*)(+=0-9_]', ' ')), ' ') as host,
        arrayStringConcat(groupUniqArray(replaceRegexpAll(replace(decodeURLComponent(http_agent), '/..', ' pathsearcherdetected '), '[\-%./!@#$?,;:&*)(+=0-9_]', ' ')), ' ') as agent,
        arrayStringConcat(groupUniqArray(replaceRegexpAll(replace(decodeURLComponent(http_query), '/..', ' pathsearcherdetected '), '[\-%./!@#$?,;:&*)(+=0-9_]', ' ')), ' ') as query,
        '{attack}' as label
    
    from dti.dti_sh_demo_log
        where (logtime >= '{start_date}' and logtime < '{end_date}')
            and hash == '{attack}'
        group by lgtime, src_ip, dst_ip
    """.format(interval=interval, attack=attack, start_date=start_date, end_date=end_date)
    
    return sql