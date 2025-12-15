import sqlite3
import os
import subprocess
import csv


def read_csv(file_path, typ='op'):
    if typ == 'op':
        rows = []
        with open(file_path, "r") as csvfile:
            read_res = csv.reader(csvfile, delimiter=",")
            OP_INDEX = 0
            for i, items in enumerate(read_res):
                if i == 0:
                    for ii, key in enumerate(items):
                        if key == 'Range':
                            OP_INDEX = ii
                            break
                else:
                    rows.append(items[OP_INDEX])
        return rows
    if typ == 'dict':
        rows = {}
        with open(file_path, "r") as csvfile:
            read_res = csv.reader(csvfile, delimiter=",")
            for i, items in enumerate(read_res):
                rows[items[0]] = items[-1]
        return rows

def read_csv_all(file_path):
    rows = []
    with open(file_path, "r") as csvfile:
        read_res = csv.reader(csvfile, delimiter=",")
        for i, items in enumerate(read_res):
            # print(i, items)
            rows.append(items)
    return rows


def nsys_stats_report(sqlite_name, method, args, nsys_rep):
    if not os.path.exists(args):
        if method == 'aten_op_list':
            method = os.path.join(os.path.dirname(__file__), f'{method}.py')
        cmd = f'rm -rf {sqlite_name}_{method}.csv;'
        # print(cmd)
        subprocess.Popen(cmd, shell=True,).wait()
        # input('Pause')
        create_sqlite = (f"nsys stats "
                        f"--report {method} "
                        f"--force-export=true "
                        f"-o {sqlite_name} "
                        f"{nsys_rep}")
        subprocess.Popen(create_sqlite, shell=True,).wait()


class Sqlite():
    def __init__(self, args):
        self.args = args
        db_name = args if isinstance(args, str) else args.d
        # 连接到SQLite数据库
        self.con = sqlite3.connect(db_name)
        # 创建一个cursor对象
        self.cur = self.con.cursor()

    def close(self):
        # 关闭cursor和连接
        self.cur.close()
        self.con.close()

    def execute(self, cmd, fetch):
        # 执行SQL查询
        self.cur.execute(cmd)
        # 获取结果
        return getattr(self.cur, fetch)()

    def table_nums(self):
        result = self.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table';", 'fetchone')
        # 打印表的数量
        print("表的数量：", result)

    def table_info(self):
        tables = self.execute("SELECT name FROM sqlite_master WHERE type='table';", 'fetchall')
        # 遍历并打印所有表的名称
        for table in tables:
            print(table[0])
            self.cur.execute(f"PRAGMA table_info({table[0]});")
            columns = self.cur.fetchall()
            print(f"表 {table[0]} 的表头：")
            for column in columns:
                print(column[1])
                print(column[2:3])

    def load_table(self, table_name):
        rows = self.execute("SELECT * FROM " + table_name + ";", 'fetchall')
        # 遍历并打印查询结果
        for row in rows:
            print(row)
        self.cur.execute(f"PRAGMA table_info({table_name});")
        columns = self.cur.fetchall()
        print(f"表 {table_name} 的表头：", [k[1] for k in columns])
        # for column in columns:
        #     print(column[1])
        #     print(column[2:3])

    def time_line(self):
        # cmd = "SELECT names.value AS name, deviceId, streamId, start, end - start FROM CUPTI_ACTIVITY_KIND_KERNEL \
        #             AS k JOIN StringIds AS names ON k.shortName = names.id;"
        # cmd = "SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL"
        # self.cur.execute(f"PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL);")
        # columns = self.cur.fetchall()
        # print(f"表 CUPTI_ACTIVITY_KIND_KERNEL 的表头：", [k[1] for k in columns])
        # rows = self.execute(cmd, 'fetchall')
        # for row in rows:
        #     print(row)
        cmd = "SELECT snames.value AS sname, start, end - start, deviceId, streamId, \
                    (globalPid >> 24) & 0x00FFFFFF AS PID, \
                    fnames.value AS fname FROM CUPTI_ACTIVITY_KIND_KERNEL AS k \
                    JOIN StringIds AS snames ON k.shortName = snames.id \
                    JOIN StringIds AS fnames ON k.demangledName = fnames.id;"
        rows = self.execute(cmd, 'fetchall')
        # 遍历并打印查询结果
        _t = 0
        for row in rows:
            now = int(row[1])
            if now < _t:
                print('timeline err:', now, _t)
                exit(0)
            # print(row[0], row[1:-1], row[-1][:50])
            _t = now
        return rows

    def time_line_nvtx(self):
        EVENT_TYPE_NVTX_DOMAIN_CREATE = 75
        EVENT_TYPE_NVTX_PUSHPOP_RANGE = 59
        EVENT_TYPE_NVTX_STARTEND_RANGE = 60
        EVENT_TYPE_NVTXT_PUSHPOP_RANGE = 70
        EVENT_TYPE_NVTXT_STARTEND_RANGE = 71

        cmd = f"""
        WITH
            domains AS (
                SELECT
                    min(start),
                    domainId AS id,
                    globalTid AS globalTid,
                    text AS name
                FROM
                    NVTX_EVENTS
                WHERE
                    eventType == {EVENT_TYPE_NVTX_DOMAIN_CREATE}
                GROUP BY 2, 3
            ),
            maxts AS(
                SELECT max(max(start), max(end)) AS m
                FROM   NVTX_EVENTS
            )
        SELECT
            ne.start AS start,
            coalesce(ne.end, (SELECT m FROM maxts)) - ne.start AS duration,
            CASE
                WHEN d.name NOT NULL AND sid.value IS NOT NULL
                    THEN d.name || ':' || sid.value
                WHEN d.name NOT NULL AND sid.value IS NULL
                    THEN d.name || ':' || ne.text
                WHEN d.name IS NULL AND sid.value NOT NULL
                    THEN sid.value
                ELSE ne.text
            END AS tag,
            CASE ne.eventType
                WHEN {EVENT_TYPE_NVTX_PUSHPOP_RANGE}
                    THEN 'PushPop'
                WHEN {EVENT_TYPE_NVTX_STARTEND_RANGE}
                    THEN 'StartEnd'
                WHEN {EVENT_TYPE_NVTXT_PUSHPOP_RANGE}
                    THEN 'PushPop'
                WHEN {EVENT_TYPE_NVTXT_STARTEND_RANGE}
                    THEN 'StartEnd'
                ELSE 'Unknown'
            END AS style
        FROM
            NVTX_EVENTS AS ne
        LEFT OUTER JOIN
            domains AS d
            ON ne.domainId == d.id
                AND (ne.globalTid & 0x0000FFFFFF000000) == (d.globalTid & 0x0000FFFFFF000000)
        LEFT OUTER JOIN
            StringIds AS sid
            ON ne.textId == sid.id
        WHERE
            ne.eventType == {EVENT_TYPE_NVTX_PUSHPOP_RANGE}
            OR
            ne.eventType == {EVENT_TYPE_NVTX_STARTEND_RANGE}
            OR
            ne.eventType == {EVENT_TYPE_NVTXT_PUSHPOP_RANGE}
            OR
            ne.eventType == {EVENT_TYPE_NVTXT_STARTEND_RANGE}
        ;
        """
        rows = self.execute(cmd, 'fetchall')
        # for row in rows:
        #     print(row)
        return rows

    def kernel_nvtx(self):
        EVENT_TYPE_NVTX_DOMAIN_CREATE = 75
        EVENT_TYPE_NVTX_PUSHPOP_RANGE = 59
        EVENT_TYPE_NVTX_STARTEND_RANGE = 60
        EVENT_TYPE_NVTXT_PUSHPOP_RANGE = 70
        EVENT_TYPE_NVTXT_STARTEND_RANGE = 71

        cmd = f"""
        WITH
            domains AS (
                SELECT
                    min(start),
                    domainId AS id,
                    globalTid AS globalTid,
                    text AS name
                FROM
                    NVTX_EVENTS
                WHERE
                    eventType == {EVENT_TYPE_NVTX_DOMAIN_CREATE}
                GROUP BY 2, 3
            ),
            maxts AS(
                SELECT max(max(start), max(end)) AS m
                FROM   NVTX_EVENTS
            )
        SELECT
            ne.start AS start,
            coalesce(ne.end, (SELECT m FROM maxts)) - ne.start AS duration,
            CASE
                WHEN d.name NOT NULL AND sid.value IS NOT NULL
                    THEN d.name || ':' || sid.value
                WHEN d.name NOT NULL AND sid.value IS NULL
                    THEN d.name || ':' || ne.text
                WHEN d.name IS NULL AND sid.value NOT NULL
                    THEN sid.value
                ELSE ne.text
            END AS tag,
            CASE ne.eventType
                WHEN {EVENT_TYPE_NVTX_PUSHPOP_RANGE}
                    THEN 'PushPop'
                WHEN {EVENT_TYPE_NVTX_STARTEND_RANGE}
                    THEN 'StartEnd'
                WHEN {EVENT_TYPE_NVTXT_PUSHPOP_RANGE}
                    THEN 'PushPop'
                WHEN {EVENT_TYPE_NVTXT_STARTEND_RANGE}
                    THEN 'StartEnd'
                ELSE 'Unknown'
            END AS style
        FROM
            NVTX_EVENTS AS ne
        LEFT OUTER JOIN
            domains AS d
            ON ne.domainId == d.id
                AND (ne.globalTid & 0x0000FFFFFF000000) == (d.globalTid & 0x0000FFFFFF000000)
        LEFT OUTER JOIN
            StringIds AS sid
            ON ne.textId == sid.id
        WHERE
            ne.eventType == {EVENT_TYPE_NVTX_PUSHPOP_RANGE}
            OR
            ne.eventType == {EVENT_TYPE_NVTX_STARTEND_RANGE}
            OR
            ne.eventType == {EVENT_TYPE_NVTXT_PUSHPOP_RANGE}
            OR
            ne.eventType == {EVENT_TYPE_NVTXT_STARTEND_RANGE}
        ;
        """
        rows = self.execute(cmd, 'fetchall')
        # for row in rows:
        #     print(row)
        return rows