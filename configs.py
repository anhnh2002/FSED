import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--skip-first', action='store_true')
    parser.add_argument('--log-dir', default='./outputs/log_terminal/02-10-nomap-clreps')
    parser.add_argument('--tb-dir', default='./outputs/log_tensorboard/02-10-nomap-clreps')
    parser.add_argument('--save-dir', default='')
    parser.add_argument('--resume', default='')
    parser.add_argument('--parallel', default='single', choices=['single', 'DP', 'DDP'])
    parser.add_argument('--device_ids', default='0,1')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--amp", action='store_true') 
    parser.add_argument('--perm-id', default=0, type=str, choices=[str(i) for i in range(5)])
    parser.add_argument('--dataset', default='MAVEN', choices=['MAVEN', 'ACE'])
    parser.add_argument('--stream-root', default='./data_incremental', type=str)
    parser.add_argument('--max_seqlen', default=120)
    parser.add_argument('--adamw_eps', default=1e-7)
    parser.add_argument('--fixed-enum', default=True, type=bool, help="whether to fix the exemplar number")
    parser.add_argument('--enum', default=1, type=int, help="When 'fixed-num' == False, indicates the the whole memory size\
                                                            when 'fixed-num' == True, indicates every class's exemplar num")
    parser.add_argument('--temperature', default=2)
    parser.add_argument('--task-num', default=5, type=int)
    parser.add_argument('--early-stop', action='store_true')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--eval_freq', type=int, default=1)

    parser.add_argument('--input-map', action='store_true', help="Whether to use input mapping, if False, use span_s to predict trigger type")
    parser.add_argument('--class-num', type=int, default=10)
    parser.add_argument('--shot-num', default=5, type=int)
    parser.add_argument('--e_weight', default=50)
    parser.add_argument('--no-replay', action='store_true')
    parser.add_argument('--period', type=int, default=10)
    parser.add_argument('--epochs', default=20, type=int) 
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--device', default="cuda:2", help='set device cuda or cpu')
    parser.add_argument('--log', action='store_true') 
    parser.add_argument('--log-name', default='temp')
    parser.add_argument('--data-root', default='./data_incremental', type=str)
    parser.add_argument('--backbone', default='bert-base-uncased', help='Feature extractor')
    parser.add_argument('--lr',type=float, default=2e-5)
    parser.add_argument('--decay', type=float, default=1e-4, help="")
    parser.add_argument('--no-freeze-bert', action='store_true')
    parser.add_argument('--dweight_loss', action='store_true')
    parser.add_argument('--alpha', type=float, default=2.0)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--distill', choices=["fd", "pd", "mul", "none"], default="mul")
    parser.add_argument('--rep-aug', choices=["none", "mean", "relative"], default="mean")
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--theta',type=float, default=6)
    # parser.add_argument('--ecl', required=True, choices=["dropout", "shuffle", "RTR", "none"])
    parser.add_argument('--cl_temp', type=float, default=0.5)
    parser.add_argument('--ucl', action='store_true')
    parser.add_argument('--cl-aug', choices=["dropout", "shuffle", "RTR", "none"])
    parser.add_argument('--sub-max', action='store_true')
    parser.add_argument('--leave-zero', action='store_true')
    parser.add_argument('--single-label', action='store_true')
    parser.add_argument('--aug-repeat-times', type=int, default=1)
    parser.add_argument('--joint-da-loss', default="none", choices=["none", "ce", "dist", "mul"])
    parser.add_argument('--tlcl', action="store_true")
    parser.add_argument('--skip-first-cl', choices=["ucl", "tlcl", "ucl+tlcl", "none"], default="none")
    parser.add_argument('--method', type=str)
    args = parser.parse_args()
    return args